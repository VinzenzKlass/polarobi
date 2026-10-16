from typing import Any, Literal

import gurobipy as gp

from polarobi import _native

type Kind = Literal["var", "constr"]


def register(model: Any, kind: Kind, start: int, count: int) -> None:
    """Add handles for `count` new elements from index `start` to gurobipy's bookkeeping."""
    pool = model._internal_vars if kind == "var" else model._constrs
    _native.register(pool, gp.Var if kind == "var" else gp.Constr, model._cmodel, start, count)
    if kind == "var":
        model._livevars += count
    else:
        model._liveconstrs += count
