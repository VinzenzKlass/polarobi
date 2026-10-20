"""
The specification of the `df.grb` namespace, written against the public gurobipy API.

Each function does what the namespace documents, row by row in plain Python, the way a gurobipy
user would write it. Wherever gurobipy has an equivalent, it is the oracle: variables keyed by
columns are added with `Model.addVars`, constraints named from keys with `Model.addConstrs`, so
names and error behaviour are gurobipy's own. Nothing here reuses polarobi code.

Polars only evaluates the caller's inputs (column names and expressions), exactly as a user's
own code would before handing values to gurobipy.
"""

from collections.abc import Sequence
from typing import Any

import gurobipy as gp
import polars as pl

from tests.harness.specs import (
    AddConstr,
    AddConstrs,
    AddVar,
    AddVars,
    GetAttr,
    IntoColumn,
    Kind,
    SetAttr,
)

type Key = tuple[Any, ...]

_SENSE = {
    "<": gp.GRB.LESS_EQUAL,
    "<=": gp.GRB.LESS_EQUAL,
    ">": gp.GRB.GREATER_EQUAL,
    ">=": gp.GRB.GREATER_EQUAL,
    "=": gp.GRB.EQUAL,
    "==": gp.GRB.EQUAL,
}


def add_var(model: gp.Model, spec: AddVar) -> pl.DataFrame:
    var = model.addVar(lb=spec.lb, ub=spec.ub, obj=spec.obj, vtype=spec.vtype, name=spec.name or "")
    model.update()
    return pl.DataFrame([_index(spec.idx, [var])])


def add_vars(model: gp.Model, spec: AddVars) -> pl.DataFrame:
    keys = spec.keys
    by = list(keys.columns) if spec.by is None else list(spec.by)
    n = keys.height
    lb = _floats(keys, spec.lb) or [0.0] * n
    ub = _floats(keys, spec.ub) or [gp.GRB.INFINITY] * n
    obj = _floats(keys, spec.obj) or [0.0] * n
    vtype = spec.vtype or gp.GRB.CONTINUOUS
    if isinstance(spec.name, pl.Expr) or not by:
        names = _names(keys, spec.name)
        variables = [
            model.addVar(lb=lb[i], ub=ub[i], obj=obj[i], vtype=vtype, name=names[i])
            for i in range(n)
        ]
    else:
        index = list(keys.select(by).iter_rows())
        named = {} if spec.name is None else {"name": spec.name}
        added = model.addVars(index, lb=lb, ub=ub, obj=obj, vtype=vtype, **named)
        variables = [added[key] for key in index]
    model.update()
    return keys.with_columns(_index(spec.idx, variables))


def add_constr(model: gp.Model, spec: AddConstr) -> pl.DataFrame:
    return add_constrs(
        model,
        AddConstrs(
            terms=spec.terms,
            by=(),
            sense=spec.sense,
            rhs=spec.rhs,
            name=spec.name,
            var=spec.var,
            coef=spec.coef,
            idx=spec.idx,
        ),
    )


def add_constrs(model: gp.Model, spec: AddConstrs) -> pl.DataFrame:
    by = list(spec.by)
    terms = spec.terms
    model.update()
    columns = model.getVars()
    coefs = _floats(terms, spec.coef)
    assert coefs is not None
    # Without key columns every term has the empty key.
    term_keys: list[Key] = list(terms.select(by).iter_rows()) if by else [()] * terms.height
    var_index: list[int] = terms.get_column(spec.var).to_list()

    if spec.rows is None:
        # Without keys all terms form one constraint, if there are any.
        row_keys = list(dict.fromkeys(term_keys))
        rows = pl.DataFrame(row_keys, schema=terms.select(by).schema, orient="row")
    else:
        rows = spec.rows
        row_keys = list(rows.select(by).iter_rows())
    # Per-row inputs belong to the rows as given; rows without terms drop out afterwards.
    senses = _senses(rows, spec.sense, len(row_keys))
    rhs = _floats(rows, spec.rhs, len(row_keys))
    assert rhs is not None
    names = _names(rows, spec.name, len(row_keys))
    if spec.rows is not None and spec.drop_empty:
        present = set(term_keys)
        keep = [key in present for key in row_keys]
        rows = rows.filter(pl.Series(keep, dtype=pl.Boolean))
        row_keys = _kept(row_keys, keep)
        senses, rhs, names = _kept(senses, keep), _kept(rhs, keep), _kept(names, keep)

    # Rows may repeat a key; each such row gets all of the key's terms. A variable repeated within
    # a row is summed left to right, as documented; gurobipy would sum repeats in an order of its
    # own, an unstable sort that depends on the other terms, which polarobi does not reproduce.
    summed: dict[Key, dict[int, float]] = {key: {} for key in row_keys}
    for key, coef, column in zip(term_keys, coefs, var_index, strict=True):
        if key in summed:
            row = summed[key]
            row[column] = row[column] + coef if column in row else coef
    lhs = {key: [(c, columns[i]) for i, c in row.items()] for key, row in summed.items()}

    before = model.NumConstrs
    if isinstance(spec.name, str) and by:
        constraints = iter(_temp_constrs(row_keys, lhs, senses, rhs))
        # gurobipy names each constraint by the generator's loop variable, so it must be the key.
        # Its stub only types the nonlinear overload of addConstrs.
        model.addConstrs((next(constraints) for _key in row_keys), name=spec.name)  # pyright: ignore[reportCallIssue, reportArgumentType]
    else:
        for i, key in enumerate(row_keys):
            model.addLConstr(gp.LinExpr(lhs[key]), senses[i], rhs[i], name=names[i])
    model.update()
    # gurobipy's tupledict keeps one constraint per repeated key, so take them in model order.
    created = model.getConstrs()[before:]
    if not by:  # no key columns to return
        return pl.DataFrame([_index(spec.idx, created)])
    return rows.with_columns(_index(spec.idx, created))


def set_attr(model: gp.Model, spec: SetAttr) -> None:
    elements = _elements(model, spec.index, spec.kind)
    values = spec.values
    if isinstance(values, str):
        values = spec.index.get_column(values)
    listed = values.to_list() if isinstance(values, pl.Series) else [values] * len(elements)
    model.setAttr(spec.attr, elements, listed)
    model.update()


def get_attr(model: gp.Model, spec: GetAttr) -> pl.Series:
    elements = _elements(model, spec.index, spec.kind)
    return pl.Series(spec.attr, model.getAttr(spec.attr, elements), dtype=spec.dtype)


def handles(model: gp.Model, index: pl.DataFrame, kind: Kind) -> list[Any]:
    return _elements(model, index, kind)


def _elements(model: gp.Model, index: pl.DataFrame, kind: Kind) -> list[Any]:
    model.update()
    pool: Sequence[Any] = model.getVars() if kind == "var" else model.getConstrs()
    return [pool[i] for i in index.get_column("idx").to_list()]


def _index(name: str, elements: Sequence[Any]) -> pl.Series:
    return pl.Series(name, [e.index for e in elements], dtype=pl.UInt32)


def _values(
    df: pl.DataFrame, value: IntoColumn | None, height: int | None = None
) -> list[Any] | None:
    """The per-row values of a scalar, column name, expression or Series input."""
    height = df.height if height is None else height
    match value:
        case None:
            return None
        case pl.Series():
            return value.to_list()
        case str():
            return df.get_column(value).to_list()
        case pl.Expr():
            column = df.select(value).to_series() if df.width else pl.select(value).to_series()
            if column.len() == 1 and height != 1:  # a literal applies to every row
                return column.to_list() * height
            return column.to_list()
        case _:
            return [value] * height


def _floats(
    df: pl.DataFrame, value: IntoColumn | None, height: int | None = None
) -> list[float] | None:
    values = _values(df, value, height)
    return None if values is None else [float(v) for v in values]


def _senses(rows: pl.DataFrame, sense: str | pl.Series | pl.Expr, height: int) -> list[str]:
    values = [sense] * height if isinstance(sense, str) else _values(rows, sense, height)
    assert values is not None
    return [_SENSE[v] for v in values]


def _names(df: pl.DataFrame, name: str | pl.Expr | None, height: int | None = None) -> list[str]:
    height = df.height if height is None else height
    if name is None:
        return [""] * height
    if isinstance(name, str):
        return [name] * height
    values = _values(df, name.cast(pl.String), height)
    assert values is not None
    # A null name leaves the element unnamed: gurobipy's default empty name.
    return ["" if v is None else str(v) for v in values]


def _kept[T](values: Sequence[T], keep: Sequence[bool]) -> list[T]:
    return [value for value, k in zip(values, keep, strict=True) if k]


def _temp_constrs(
    keys: Sequence[Key],
    lhs: dict[Key, list[tuple[float, gp.Var]]],
    senses: Sequence[str],
    rhs: Sequence[float],
) -> list[gp.TempConstr]:
    constraints: list[gp.TempConstr] = []
    for key, sense, bound in zip(keys, senses, rhs, strict=True):
        expr = gp.LinExpr(lhs[key])
        if sense == gp.GRB.LESS_EQUAL:
            constraints.append(expr <= bound)
        elif sense == gp.GRB.GREATER_EQUAL:
            constraints.append(expr >= bound)
        else:
            constraints.append(expr == bound)
    return constraints
