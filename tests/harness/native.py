# Type checkers cannot see the runtime-registered `df.grb` namespace. This module is the one place
# the harness calls it, so the rest of the harness stays fully typed.
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Apply specs through polarobi's `df.grb` namespace."""

from typing import Any

import gurobipy as gp
import polarobi as polarobi  # registers df.grb
import polars as pl

from tests.harness.specs import AddConstrs, AddVar, AddVars, GetAttr, Kind, SetAttr


def add_var(model: gp.Model, spec: AddVar) -> pl.DataFrame:
    return polarobi.add_var(
        model, lb=spec.lb, ub=spec.ub, obj=spec.obj, vtype=spec.vtype, name=spec.name, idx=spec.idx
    )


def add_vars(model: gp.Model, spec: AddVars) -> pl.DataFrame:
    result: pl.DataFrame = spec.keys.grb.add_vars(
        model,
        by=spec.by,
        lb=spec.lb,
        ub=spec.ub,
        obj=spec.obj,
        vtype=spec.vtype,
        name=spec.name,
        idx=spec.idx,
    )
    return result


def add_constrs(model: gp.Model, spec: AddConstrs) -> pl.DataFrame:
    result: pl.DataFrame = spec.terms.grb.add_constrs(
        model,
        spec.rows,
        by=spec.by,
        sense=spec.sense,
        rhs=spec.rhs,
        name=spec.name,
        var=spec.var,
        coef=spec.coef,
        drop_empty=spec.drop_empty,
        idx=spec.idx,
    )
    return result


def set_attr(model: gp.Model, spec: SetAttr) -> None:
    spec.index.grb.set(model, spec.attr, spec.values)


def get_attr(model: gp.Model, spec: GetAttr) -> pl.Series:
    result: pl.DataFrame = spec.index.grb.get(model, spec.attr, dtype=spec.dtype)
    return result.get_column(spec.attr)


def handles(model: gp.Model, index: pl.DataFrame, kind: Kind) -> list[Any]:
    result: list[Any] = index.grb.handles(model, kind=kind)
    return result
