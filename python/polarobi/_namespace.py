from collections.abc import Sequence
from typing import Any, Literal, overload

import gurobipy as gp
import polars as pl

from polarobi import _api
from polarobi._api import IntoColumn, Sense, VType


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
        vtype: VType | None = None,
        name: str | pl.Expr | None = None,
        idx: str = "idx",
    ) -> pl.DataFrame:
        """
        Add one variable per row, returning this frame with the variables' column index `idx`.
        Like `Model.addVars`, raises `KeyError` for duplicate keys, but before adding anything.

        :param model: The Gurobi model to add variables to.
        :param by: Columns to use as keys for variable names.
        :param lb: Lower bound for variables.
        :param ub: Upper bound for variables.
        :param obj: Objective coefficient for variables.
        :param vtype: Variable type to create: `C` (continuous), `B` (binary),
            `I` (integer), `S` (semi-continuous) or `N` (semi-integer).
        :param name: Name for variables.
        :param idx: Column name for variable indices.
        :return: DataFrame with added variables.
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

        :param model: The Gurobi model to add constraints to.
        :param rows: DataFrame with one row per constraint containing the keys `by`. Defaults to
            distinct keys of the terms in order of appearance.
        :param by: Columns to use as keys for grouping terms into constraints.
        :param sense: Constraint sense: `<` or `<=` (less than or equal), `>` or `>=`
            (greater than or equal), or `=` or `==` (equal).
        :param rhs: Right-hand side for constraints.
        :param name: Name for constraints.
        :param var: Column name for variable indices in this frame.
        :param coef: Coefficient for each term.
        :param drop_empty: Whether to drop rows in `rows` that have no matching terms.
        :param idx: Column name for constraint indices.
        :return: DataFrame with added constraints.
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
        Return this frame with `attr` (e.g. "X", "Pi", "UB") of its variables or constraints added.
        Values at Gurobi's infinity read as `inf`, as in gurobipy.

        :param model: The Gurobi model to read the attribute from.
        :param attr: Name of the Gurobi attribute to retrieve.
        :param idx: Column name for variable or constraint indices.
        :param dtype: Polars DType for the returned attribute column (`pl.Float64` or `pl.Int32`).
        :return: DataFrame with the attribute column added.
        """
        values = _api.get_attr(model, attr, self._df.get_column(idx), dtype=dtype)
        return self._df.with_columns(values)

    def set(
        self, model: gp.Model, attr: str, values: pl.Series | str | float, *, idx: str = "idx"
    ) -> None:
        """
        Set `attr` of this frame's variables or constraints.
        Integer values use Gurobi's int attribute API.

        :param model: The Gurobi model to set the attribute on.
        :param attr: Name of the Gurobi attribute to set.
        :param values: Attribute values as a scalar, a Series, or a column name.
        :param idx: Column name for variable or constraint indices.
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
        """
        Get the gurobipy `Var` or `Constr` handles of this frame's rows.

        :param model: The Gurobi model the variables or constraints belong to.
        :param idx: Column name for variable or constraint indices.
        :param kind: Handle type to return: `var` (variable) or `constr` (constraint).
        :return: List of Gurobi handles.
        """
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
        Return a gurobipy `tupledict` of this frame's handles keyed like `Model.addVars` keys it:
        by the values of `by` for one column, else by tuples.

        :param model: The Gurobi model the variables or constraints belong to.
        :param by: Column or columns to use as keys.
        :param idx: Column name for variable or constraint indices.
        :param kind: Handle type to return: `var` (variable) or `constr` (constraint).
        :return: `tupledict` mapping keys to Gurobi handles.
        """
        return _api.tupledict(model, self._df, by, idx=idx, kind=kind)
