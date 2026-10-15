"""
Implementation of the `df.grb` namespace, documented there.

Data goes from Arrow memory into the Gurobi C API in one call per family, without materializing
Python objects per row. The created columns and rows are then registered with gurobipy as regular
`gp.Var` / `gp.Constr` handles, so the model stays fully usable through the gurobipy API:
`getVars`, `getVarByName`, expressions mixing both kinds of variables, `remove`, `copy`, ...

Where gurobipy has an equivalent call, the result matches it exactly, down to names and signed
zeros; `tests/test_equivalence.py` checks that against plain gurobipy.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import gurobipy as gp
import polars as pl

from polarobi import _native

type IntoColumn = pl.Series | pl.Expr | str | float | int
type Sense = Literal["<", "<=", ">", ">=", "=", "=="]

_SENSE = {
    "<": ord("<"),
    "<=": ord("<"),
    ">": ord(">"),
    ">=": ord(">"),
    "=": ord("="),
    "==": ord("="),
}
_ROW = "__polarobi_row"
_INDEX_TYPES = (pl.Int32, pl.UInt32)
# gurobipy reports values at or beyond Gurobi's infinity as Python infinities.
_INFINITY = gp.GRB.INFINITY

# gurobipy's stubs hide the handle constructors `Var(cmodel, colno)` and `Constr(cmodel, rowno)`.
_Var = cast(Callable[[object, int], gp.Var], gp.Var)
_Constr = cast(Callable[[object, int], gp.Constr], gp.Constr)


def add_vars(
    model: gp.Model,
    keys: pl.DataFrame,
    *,
    by: Sequence[str] | None = None,
    lb: IntoColumn | None = None,
    ub: IntoColumn | None = None,
    obj: IntoColumn | None = None,
    vtype: str | None = None,
    name: str | pl.Expr | None = None,
    idx: str = "idx",
) -> pl.DataFrame:
    by = keys.columns if by is None else list(by)
    if by and keys.select(by).is_duplicated().any():
        # gurobipy's Model.addVars raises this too, though only after adding the variables.
        raise KeyError("Duplicate keys in add_vars()")
    _sync(model)
    n, start = keys.height, len(model._internal_vars)
    _native.add_vars(
        model._cmodel,
        n,
        obj=_optional_column(keys, obj, pl.Float64),
        lb=_optional_column(keys, lb, pl.Float64),
        ub=_optional_column(keys, ub, pl.Float64),
        vtype=None if vtype is None else pl.repeat(ord(vtype), n, dtype=pl.UInt8, eager=True),
        names=_names(keys, name, by),
    )
    cmodel = model._cmodel
    model._internal_vars.extend([_Var(cmodel, i) for i in range(start, start + n)])
    model._livevars += n
    return keys.with_columns(_index(start, n).alias(idx))


def add_constrs(
    model: gp.Model,
    terms: pl.DataFrame,
    rows: pl.DataFrame | None = None,
    *,
    by: str | Sequence[str],
    sense: Sense | pl.Series | pl.Expr,
    rhs: IntoColumn = 0.0,
    name: str | pl.Expr | None = None,
    var: str = "idx",
    coef: IntoColumn = "coef",
    drop_empty: bool = True,
    idx: str = "idx",
) -> pl.DataFrame:
    by = [by] if isinstance(by, str) else list(by)
    if not by:
        return _add_constr_over_all(model, terms, rows, sense, rhs, name, var, coef, idx)
    _sync(model)
    terms = terms.select(
        *by, pl.col(var).alias(_ROW + "var"), _expr(coef).cast(pl.Float64).alias(_ROW + "coef")
    )
    if rows is None:
        rows = terms.select(by).unique(maintain_order=True)
    # Per-row inputs are evaluated against `rows` as given, so a Series lines up with it, and
    # are then dropped along with rows that have no terms. gurobipy turns a -0.0 bound into 0.0
    # (polars simplifies `x + 0.0` to `x`, so that does not work here).
    names = _names(rows, name, by)
    bound = _column(rows, rhs, pl.Float64)
    rows = rows.with_columns(
        bound.to_frame()
        .select(pl.when(bound == 0).then(0.0).otherwise(bound))
        .to_series()
        .alias(_ROW + "rhs"),
        _sense(rows, sense).alias(_ROW + "sense"),
        *([] if names is None else [names.alias(_ROW + "name")]),
    )
    if drop_empty:
        rows = rows.join(terms.select(by), on=by, how="semi", maintain_order="left")
    rows = rows.with_row_index(_ROW)
    terms = terms.join(rows.select(*by, _ROW), on=by, how="inner", maintain_order="left").sort(
        _ROW, maintain_order=True
    )
    cbeg = terms.get_column(_ROW).search_sorted(rows.get_column(_ROW), side="left")
    n, start = rows.height, len(model._constrs)
    _native.add_constrs(
        model._cmodel,
        cbeg.cast(pl.UInt64),
        _indices(terms.get_column(_ROW + "var")),
        terms.get_column(_ROW + "coef"),
        rows.get_column(_ROW + "sense"),
        rows.get_column(_ROW + "rhs"),
        names=None if names is None else rows.get_column(_ROW + "name"),
    )
    cmodel = model._cmodel
    model._constrs.extend([_Constr(cmodel, i) for i in range(start, start + n)])
    model._liveconstrs += n
    hidden = [c for c in rows.columns if c.startswith(_ROW)]
    return rows.drop(hidden).with_columns(_index(start, n).alias(idx))


def _add_constr_over_all(
    model: gp.Model,
    terms: pl.DataFrame,
    rows: pl.DataFrame | None,
    sense: Sense | pl.Series | pl.Expr,
    rhs: IntoColumn,
    name: str | pl.Expr | None,
    var: str,
    coef: IntoColumn,
    idx: str,
) -> pl.DataFrame:
    """Without key columns, all terms form a single constraint."""
    if rows is not None:
        raise ValueError("rows need key columns: pass `by`")
    key = "__polarobi_all"
    terms = terms.with_columns(pl.lit(True).alias(key))
    # A plain name, not `name[True]`, as gurobipy names a constraint without keys.
    name = pl.lit(name) if isinstance(name, str) else name
    added = add_constrs(
        model, terms, by=key, sense=sense, rhs=rhs, name=name, var=var, coef=coef, idx=idx
    )
    return added.drop(key)


def get_attr(
    model: gp.Model,
    attr: str,
    index: pl.Series,
    *,
    dtype: type[pl.Float64] | type[pl.Int32] = pl.Float64,
) -> pl.Series:
    """Read `attr` (e.g. "X", "Pi", "UB") for the variables or constraints at `index`."""
    _sync(model)
    values = pl.Series(
        attr, _native.get_attr(model._cmodel, attr, _indices(index), int=dtype == pl.Int32)
    )
    if dtype == pl.Int32:
        return values
    infinite = pl.when(pl.col(attr) >= _INFINITY).then(float("inf"))
    infinite = infinite.when(pl.col(attr) <= -_INFINITY).then(float("-inf"))
    return values.to_frame().select(infinite.otherwise(pl.col(attr)).alias(attr)).to_series()


def set_attr(model: gp.Model, attr: str, index: pl.Series, values: pl.Series | float) -> None:
    """Set `attr` for the variables or constraints at `index`; integer values use the int API."""
    _sync(model)
    if not isinstance(values, pl.Series):
        dtype = pl.Int32 if isinstance(values, int) and not isinstance(values, bool) else pl.Float64
        values = pl.repeat(values, index.len(), dtype=dtype, eager=True)
    values = values.cast(pl.Int32 if values.dtype.is_integer() else pl.Float64)
    _native.set_attr(model._cmodel, attr, _indices(index), values)


def handles(
    model: gp.Model, index: pl.Series, *, kind: Literal["var", "constr"] = "var"
) -> list[Any]:
    """The gurobipy `Var` / `Constr` handles at `index`."""
    _sync(model)
    pool = model._internal_vars if kind == "var" else model._constrs
    return [pool[i] for i in index]


def tupledict(
    model: gp.Model,
    df: pl.DataFrame,
    by: str | Sequence[str],
    *,
    idx: str = "idx",
    kind: Literal["var", "constr"] = "var",
) -> gp.tupledict[Any, Any]:
    """A gurobipy `tupledict` keyed like `addVars` keys it: scalars for one key, else tuples."""
    keys = df.get_column(by) if isinstance(by, str) else df.select(by).iter_rows()
    return gp.tupledict(zip(keys, handles(model, df.get_column(idx), kind=kind), strict=True))


def _sync(model: gp.Model) -> None:
    # Pending deletions shift Gurobi's indices on update, and positions in gurobipy's handle
    # lists until then; every index here means the position after the update.
    if model._has_pending_deletions():
        model.update()


def _index(start: int, n: int) -> pl.Expr:
    return pl.int_range(start, start + n, dtype=pl.UInt32)


def _indices(index: pl.Series) -> pl.Series:
    return index if isinstance(index.dtype, _INDEX_TYPES) else index.cast(pl.Int32)


def _expr(value: IntoColumn) -> pl.Expr:
    if isinstance(value, pl.Expr):
        return value
    if isinstance(value, str):
        return pl.col(value)
    if isinstance(value, pl.Series):
        return pl.lit(value)
    return pl.lit(float(value))


def _optional_column(
    df: pl.DataFrame, value: IntoColumn | None, dtype: pl.DataType | type[pl.DataType]
) -> pl.Series | None:
    return None if value is None else _column(df, value, dtype)


def _column(
    df: pl.DataFrame, value: IntoColumn, dtype: pl.DataType | type[pl.DataType]
) -> pl.Series:
    """Per-row values of `df`; scalars and literal expressions apply to every row."""
    if isinstance(value, pl.Series):
        if value.len() != df.height:
            raise ValueError(f"expected {df.height} values, got {value.len()}")
        return value.cast(dtype)
    column = _ROW + "value"
    # `with_columns`, unlike `select`, broadcasts a literal to the frame's height.
    return df.with_columns(_expr(value).cast(dtype).alias(column)).get_column(column)


def _sense(rows: pl.DataFrame, sense: Sense | pl.Series | pl.Expr) -> pl.Series:
    if isinstance(sense, str):
        return pl.repeat(_SENSE[sense], rows.height, dtype=pl.UInt8, eager=True)
    column = sense if isinstance(sense, pl.Series) else _column(rows, sense, pl.String)
    if column.len() != rows.height:
        raise ValueError(f"expected {rows.height} senses, got {column.len()}")
    return column.replace_strict(_SENSE, return_dtype=pl.UInt8)


def _names(df: pl.DataFrame, name: str | pl.Expr | None, by: Sequence[str]) -> pl.Series | None:
    if name is None:
        return None
    if isinstance(name, pl.Expr):
        # A null name leaves the element unnamed, like gurobipy's default empty name.
        return _column(df, name, pl.String).fill_null("")
    if not by:
        return pl.repeat(name, df.height, dtype=pl.String, eager=True)
    schema = df.schema
    keys = pl.concat_str([_key_text(k, schema[k]) for k in by], separator=",")
    return df.select(pl.concat_str(pl.lit(f"{name}["), keys, pl.lit("]"))).to_series()


def _key_text(column: str, dtype: pl.DataType) -> pl.Expr:
    """A key as gurobipy writes it into names: `str()` of the Python value."""
    value = pl.col(column)
    if dtype == pl.Null:
        return pl.lit("None")
    if dtype == pl.Boolean:
        text = pl.when(value).then(pl.lit("True")).otherwise(pl.lit("False"))
    elif dtype.is_integer() or isinstance(dtype, (pl.String, pl.Date)):
        text = value.cast(pl.String)
    elif dtype.is_float():
        # Python's shortest float repr ("1e-07", "1e+16") has no polars equivalent.
        text = value.map_elements(repr, return_dtype=pl.String, skip_nulls=True)
    else:
        raise TypeError(f"key column {column!r} has unsupported type {dtype} for names")
    return pl.when(value.is_null()).then(pl.lit("None")).otherwise(text)
