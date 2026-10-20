"""
Hypothesis strategies for `df.grb` calls.

Inputs vary in everything the namespace accepts: key dtypes (with nulls and awkward values),
every input form (scalar, column name, expression, literal expression, Series), index dtypes, and
the memory layout of the frames (several chunks, or a slice at a non-zero offset), since the
native side reads the Arrow buffers directly.

Inputs stay within what gurobipy itself accepts, so both sides succeed; error behaviour is tested
separately. In particular keys are unique (gurobipy rejects duplicate keys) and strings are ASCII
(gurobipy cannot read back the names it writes for non-ASCII keys).
"""

import datetime
import string
from typing import Any

import polars as pl
from hypothesis import strategies as st

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

_SENSES = ("<", "<=", ">", ">=", "=", "==")
_VTYPES = (None, "C", "B", "I", "S", "N")
_INDEX_DTYPES = (pl.UInt32, pl.Int32, pl.Int64, pl.UInt64)
_ASCII = string.ascii_letters + string.digits + " ,[]_-.:"

type DType = type[pl.DataType]

_KEY_VALUES: dict[DType, st.SearchStrategy[Any]] = {
    pl.String: st.text(alphabet=_ASCII, max_size=6),
    pl.Int64: st.integers(-(2**63), 2**63 - 1),
    pl.Int32: st.integers(-(2**31), 2**31 - 1),
    pl.UInt32: st.integers(0, 2**32 - 1),
    pl.Boolean: st.booleans(),
    pl.Float64: st.floats(allow_nan=False, allow_infinity=False)
    | st.sampled_from([0.0, -0.0, 1e-7, 1.5e-5, 1e16, 123456789.0]),
    pl.Date: st.dates(datetime.date(1900, 1, 1), datetime.date(2100, 12, 31)),
}

_FINITE = st.floats(-1e9, 1e9, allow_nan=False) | st.sampled_from([0.0, -0.0, 1e-12, 0.1, 1 / 3])


@st.composite
def layouts(draw: st.DrawFn, df: pl.DataFrame) -> pl.DataFrame:
    """The same frame, contiguous, split into chunks, or a slice at a non-zero offset."""
    match draw(st.sampled_from(["contiguous", "chunked", "sliced"])):
        case "chunked" if df.height > 1:
            cut = draw(st.integers(1, df.height - 1))
            return pl.concat([df.slice(0, cut), df.slice(cut)], rechunk=False)
        case "sliced":
            # Rechunked first, so the slice is a view at a non-zero offset into one buffer.
            pad = draw(st.integers(1, 3))
            return pl.concat([df.clear(pad), df]).rechunk().slice(pad)
        case _:
            return df.rechunk()


@st.composite
def key_frames(draw: st.DrawFn, max_rows: int = 10) -> pl.DataFrame:
    dtypes = draw(st.lists(st.sampled_from(list(_KEY_VALUES)), min_size=1, max_size=3))
    columns = [f"k{i}" for i in range(len(dtypes))]
    values = st.tuples(*(_KEY_VALUES[d] | st.none() for d in dtypes))
    # Unique as gurobipy compares keys: by Python equality, so 0.0 and -0.0 collide.
    rows = draw(st.lists(values, max_size=max_rows, unique_by=lambda row: row))
    return pl.DataFrame(rows, schema=dict(zip(columns, dtypes, strict=True)), orient="row")


@st.composite
def into_columns(
    draw: st.DrawFn, df: pl.DataFrame, column: str, values: st.SearchStrategy[float]
) -> tuple[pl.DataFrame, IntoColumn | None]:
    """Some input form for per-row values, with the column it refers to added to `df`."""
    drawn = draw(st.lists(values, min_size=df.height, max_size=df.height))
    data = pl.Series(column, drawn, dtype=pl.Float64)
    match draw(st.sampled_from(["none", "scalar", "column", "expr", "literal", "series"])):
        case "none":
            return df, None
        case "scalar":
            return df, draw(values)
        case "column":
            return df.with_columns(data), column
        case "expr":
            return df.with_columns(data), pl.col(column) * 2
        case "literal":
            return df, pl.lit(draw(values))
        case _:
            return df, data


def _names(keys: list[str]) -> st.SearchStrategy[str | pl.Expr | None]:
    # Not pl.format: polars 2.0.0 panics on it for filtered multi-chunk string columns.
    exprs = [pl.concat_str(pl.lit("n"), pl.col(k).cast(pl.String)) for k in keys]
    return (
        st.none() | st.from_regex(r"[a-z]{1,3}", fullmatch=True) | st.sampled_from(exprs or [None])
    )


@st.composite
def add_var_specs(draw: st.DrawFn) -> AddVar:
    return AddVar(
        lb=draw(st.sampled_from([float("-inf"), 0.0, -0.0, -5.0, 1.0])),
        ub=draw(st.sampled_from([float("inf"), 1.0, 10.0, 1e6])),
        obj=draw(_FINITE),
        vtype=draw(st.sampled_from(["C", "B", "I", "S", "N"])),
        name=draw(st.none() | st.from_regex(r"[a-z]{1,6}", fullmatch=True)),
        idx=draw(st.sampled_from(["idx", "col"])),
    )


@st.composite
def add_vars_specs(draw: st.DrawFn) -> AddVars:
    keys = draw(key_frames())
    key_columns = keys.columns
    by: tuple[str, ...] | None = None
    if draw(st.booleans()):
        by = tuple(draw(st.permutations(key_columns)))  # all key columns, any order
    bounds = st.sampled_from([float("-inf"), 0.0, -5.0, 1.0])
    keys, lb = draw(into_columns(keys, "lb_", bounds))
    keys, ub = draw(into_columns(keys, "ub_", st.sampled_from([float("inf"), 1.0, 10.0, 1e6])))
    keys, obj = draw(into_columns(keys, "obj_", _FINITE))
    # With `by` None, the value columns join the keys; still unique, as the key columns are.
    return AddVars(
        keys=draw(layouts(keys)),
        by=by,
        lb=lb,
        ub=ub,
        obj=obj,
        vtype=draw(st.sampled_from(_VTYPES)),
        name=draw(_names(key_columns)),
        idx=draw(st.sampled_from(["idx", "col"])),
    )


@st.composite
def add_constr_specs(draw: st.DrawFn, num_vars: int) -> AddConstr:
    rows_ = draw(st.lists(st.tuples(st.integers(0, num_vars - 1), _FINITE), max_size=20))
    var = draw(st.sampled_from(["idx", "v"]))
    terms = pl.DataFrame(rows_, schema={var: pl.Int64, "coef": pl.Float64}, orient="row")
    terms = terms.with_columns(pl.col(var).cast(draw(st.sampled_from(_INDEX_DTYPES))))
    coef: IntoColumn = draw(
        st.sampled_from(["coef", 1.0, -2.5]) | st.just(pl.col("coef") * 3) | st.just(pl.lit(0.5))
    )
    return AddConstr(
        terms=draw(layouts(terms)),
        sense=draw(st.sampled_from(_SENSES)),
        rhs=draw(_FINITE),
        name=draw(st.none() | st.from_regex(r"[a-z]{1,6}", fullmatch=True)),
        var=var,
        coef=coef,
        idx=draw(st.sampled_from(["idx", "row"])),
    )


@st.composite
def add_constrs_specs(draw: st.DrawFn, num_vars: int) -> AddConstrs:
    by = tuple(f"g{i}" for i in range(draw(st.integers(0, 2))))
    group = st.sampled_from(["a", "b", "c", "d"])
    term_values = st.tuples(*(group for _ in by), st.integers(0, num_vars - 1), _FINITE)
    rows_ = draw(st.lists(term_values, max_size=20))
    var = draw(st.sampled_from(["idx", "v"]))
    terms = pl.DataFrame(
        rows_,
        schema={**dict.fromkeys(by, pl.String), var: pl.Int64, "coef": pl.Float64},
        orient="row",
    )
    terms = terms.with_columns(pl.col(var).cast(draw(st.sampled_from(_INDEX_DTYPES))))
    coef: IntoColumn = draw(
        st.sampled_from(["coef", 1.0, -2.5]) | st.just(pl.col("coef") * 3) | st.just(pl.lit(0.5))
    )

    rows: pl.DataFrame | None = None
    sense: Any = draw(st.sampled_from(_SENSES))
    rhs: IntoColumn = draw(_FINITE)
    if by and draw(st.booleans()):  # explicit rows need key columns
        # Keys may repeat: gurobipy adds a constraint per row, with the same name.
        keys = draw(st.lists(st.tuples(*(group for _ in by)), max_size=8))
        rows = pl.DataFrame(keys, schema=dict.fromkeys(by, pl.String), orient="row")
        rows, rhs_ = draw(into_columns(rows, "r", _FINITE))
        rhs = 0.0 if rhs_ is None else rhs_
        senses = pl.Series(
            "s",
            draw(st.lists(st.sampled_from(_SENSES), min_size=rows.height, max_size=rows.height)),
            dtype=pl.String,
        )
        rows = rows.with_columns(senses)
        sense = draw(st.sampled_from([sense, pl.col("s"), senses]))
        rows = draw(layouts(rows))
    elif draw(st.booleans()):
        rhs = pl.lit(draw(_FINITE))

    return AddConstrs(
        terms=draw(layouts(terms)),
        by=by,
        sense=sense,
        rows=rows,
        rhs=rhs,
        name=draw(_names(list(by))),
        var=var,
        coef=coef,
        drop_empty=draw(st.booleans()),
        idx=draw(st.sampled_from(["idx", "row"])),
    )


@st.composite
def index_frames(draw: st.DrawFn, count: int) -> pl.DataFrame:
    """Element indices below `count`, repeats allowed, in any integer dtype."""
    values = draw(st.lists(st.integers(0, count - 1), max_size=8))
    dtype = draw(st.sampled_from(_INDEX_DTYPES))
    return pl.DataFrame({"idx": values}).with_columns(pl.col("idx").cast(dtype))


_SETTABLE: dict[Kind, list[tuple[str, st.SearchStrategy[Any]]]] = {
    "var": [
        ("LB", st.sampled_from([0.0, -1.0, 2.5, float("-inf")])),
        ("UB", st.sampled_from([1.0, 7.0, float("inf")])),
        ("Obj", _FINITE),
        ("BranchPriority", st.integers(-5, 5)),
    ],
    "constr": [("RHS", _FINITE)],
}
_READABLE: dict[Kind, list[tuple[str, type[pl.Float64] | type[pl.Int32]]]] = {
    "var": [
        ("LB", pl.Float64),
        ("UB", pl.Float64),
        ("Obj", pl.Float64),
        ("BranchPriority", pl.Int32),
    ],
    "constr": [("RHS", pl.Float64)],
}


@st.composite
def set_attr_specs(draw: st.DrawFn, kind: Kind, count: int) -> SetAttr:
    attr, values = draw(st.sampled_from(_SETTABLE[kind]))
    index = draw(layouts(draw(index_frames(count))))
    column = pl.Series("val", draw(st.lists(values, min_size=index.height, max_size=index.height)))
    match draw(st.sampled_from(["scalar", "series", "column"])):
        case "scalar":
            return SetAttr(kind, attr, index, draw(values))
        case "series":
            return SetAttr(kind, attr, index, column)
        case _:
            return SetAttr(kind, attr, index.with_columns(column), "val")


@st.composite
def get_attr_specs(draw: st.DrawFn, kind: Kind, count: int) -> GetAttr:
    attr, dtype = draw(st.sampled_from(_READABLE[kind]))
    return GetAttr(kind, attr, draw(layouts(draw(index_frames(count)))), dtype)
