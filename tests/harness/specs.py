"""
One call to a `df.grb` method, described as data.

The equivalence tests draw a spec once and apply it to both sides: through polarobi on one
model (`harness.native`) and through plain gurobipy on another (`harness.reference`).
"""

from dataclasses import dataclass
from typing import Literal

import polars as pl

type IntoColumn = pl.Series | pl.Expr | str | float | int
type Sense = Literal["<", "<=", ">", ">=", "=", "=="]
type Kind = Literal["var", "constr"]


@dataclass(frozen=True, slots=True)
class AddVars:
    keys: pl.DataFrame
    by: tuple[str, ...] | None = None
    lb: IntoColumn | None = None
    ub: IntoColumn | None = None
    obj: IntoColumn | None = None
    vtype: str | None = None
    name: str | pl.Expr | None = None
    idx: str = "idx"


@dataclass(frozen=True, slots=True)
class AddConstrs:
    terms: pl.DataFrame
    by: tuple[str, ...]
    sense: Sense | pl.Series | pl.Expr
    rows: pl.DataFrame | None = None
    rhs: IntoColumn = 0.0
    name: str | pl.Expr | None = None
    var: str = "idx"
    coef: IntoColumn = "coef"
    drop_empty: bool = True
    idx: str = "idx"


@dataclass(frozen=True, slots=True)
class SetAttr:
    kind: Kind
    attr: str
    index: pl.DataFrame
    """Frame with the elements' `idx` column, plus any column `values` names."""
    values: pl.Series | str | float | int


@dataclass(frozen=True, slots=True)
class GetAttr:
    kind: Kind
    attr: str
    index: pl.DataFrame
    dtype: type[pl.Float64] | type[pl.Int32] = pl.Float64
