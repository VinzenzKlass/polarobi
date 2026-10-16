from collections.abc import Sequence
from typing import Any, Literal, overload

import gurobipy as gp
import polars as pl

from polarobi import _api
from polarobi._api import IntoColumn, Sense


@pl.api.register_dataframe_namespace("grb")
class GurobiFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def add_vars(
        self,
        model: gp.Model,
        *,
        by: Sequence[str] | None = None,
        lb: IntoColumn | None = None,
        ub: IntoColumn | None = None,
        obj: IntoColumn | None = None,
        vtype: str | None = None,
        name: str | pl.Expr | None = None,
        idx: str = "idx",
    ) -> pl.DataFrame:
        """
        Add one variable per row, returning this frame with the variables' column index `idx`.

        `lb`, `ub` and `obj` take a scalar, a column name, an expression or a Series; omitted they
        keep Gurobi's defaults (0, infinity, 0). A `name` string names variables gurobipy-style
        from the key columns `by` (default: all columns), e.g. `x[a,b]`, formatting keys as
        gurobipy does (`True`, `1e-07`, `None` for nulls); an expression names them freely, a
        null name leaves a variable unnamed.

        Like `Model.addVars`, raises `KeyError` for duplicate keys, but before adding anything.
        """
        return _api.add_vars(
            model, self._df, by=by, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name, idx=idx
        )

    def add_constrs(
        self,
        model: gp.Model,
        rows: pl.DataFrame | None = None,
        *,
        by: str | Sequence[str],
        sense: Sense | pl.Series | pl.Expr,
        rhs: IntoColumn = 0.0,
        name: str | pl.Expr | None = None,
        var: str = "idx",
        coef: IntoColumn = 1.0,
        drop_empty: bool = True,
        idx: str = "idx",
    ) -> pl.DataFrame:
        """
        Add one linear constraint per group of this frame's terms, `sum(coef * x[var]) <sense> rhs`.

        This frame holds one row per term: the group keys `by` and the variable index `var`.
        `coef` is a scalar, a column name, an expression or a Series; it defaults to 1, as in
        `quicksum`. `rows` holds one row per constraint with the keys `by`, against which `sense`,
        `rhs` and `name` are evaluated; it defaults to the distinct keys of the terms, in order of
        appearance. Constraints keep the order of `rows` and terms their order in this frame, so
        equal inputs build identical models. Rows without terms are dropped unless `drop_empty`
        is False; a Series for `sense` or `rhs` lines up with `rows` as given. Rows may repeat a
        key, each then gets all of its terms, as with `Model.addConstrs`. A null key is a key
        like any other: its terms form one constraint, named with `None` as gurobipy names a
        `None` key. Without key columns (`by=[]`), all terms form a single constraint.

        A variable repeated within a row has its coefficients summed left to right, and a
        right-hand side of -0.0 becomes 0.0, as in gurobipy.

        Returns `rows` with the constraints' row index `idx`.
        """
        return _api.add_constrs(
            model,
            self._df,
            rows,
            by=by,
            sense=sense,
            rhs=rhs,
            name=name,
            var=var,
            coef=coef,
            drop_empty=drop_empty,
            idx=idx,
        )

    def get(
        self,
        model: gp.Model,
        attr: str,
        *,
        idx: str = "idx",
        dtype: type[pl.Float64] | type[pl.Int32] = pl.Float64,
    ) -> pl.DataFrame:
        """
        This frame with `attr` (e.g. "X", "Pi", "UB") of its variables or constraints added.
        Values at Gurobi's infinity read as `inf`, as in gurobipy.
        """
        values = _api.get_attr(model, attr, self._df.get_column(idx), dtype=dtype)
        return self._df.with_columns(values)

    def set(
        self, model: gp.Model, attr: str, values: pl.Series | str | float, *, idx: str = "idx"
    ) -> None:
        """
        Set `attr` of this frame's variables or constraints. `values` is a scalar, a Series or a
        column name; integer values use Gurobi's int attribute API.
        """
        if isinstance(values, str):
            values = self._df.get_column(values)
        _api.set_attr(model, attr, self._df.get_column(idx), values)

    @overload
    def handles(
        self, model: gp.Model, *, idx: str = "idx", kind: Literal["var"] = "var"
    ) -> list[gp.Var]: ...
    @overload
    def handles(
        self, model: gp.Model, *, idx: str = "idx", kind: Literal["constr"]
    ) -> list[gp.Constr]: ...
    def handles(
        self, model: gp.Model, *, idx: str = "idx", kind: Literal["var", "constr"] = "var"
    ) -> list[gp.Var] | list[gp.Constr]:
        """The gurobipy `Var` / `Constr` handles of this frame's rows."""
        return _api.handles(model, self._df.get_column(idx), kind=kind)

    @overload
    def tupledict(
        self,
        model: gp.Model,
        by: str | Sequence[str],
        *,
        idx: str = "idx",
        kind: Literal["var"] = "var",
    ) -> gp.tupledict[Any, gp.Var]: ...
    @overload
    def tupledict(
        self, model: gp.Model, by: str | Sequence[str], *, idx: str = "idx", kind: Literal["constr"]
    ) -> gp.tupledict[Any, gp.Constr]: ...
    def tupledict(
        self,
        model: gp.Model,
        by: str | Sequence[str],
        *,
        idx: str = "idx",
        kind: Literal["var", "constr"] = "var",
    ) -> gp.tupledict[Any, gp.Var] | gp.tupledict[Any, gp.Constr]:
        """
        A gurobipy `tupledict` of this frame's handles keyed like `addVars` keys it: by the
        values of `by` for one column, else by tuples.
        """
        return _api.tupledict(model, self._df, by, idx=idx, kind=kind)
