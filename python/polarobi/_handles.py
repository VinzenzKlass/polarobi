from typing import Any, Literal

import gurobipy as gp

from polarobi import _native

type Kind = Literal["var", "constr"]


def target(model: Any, kind: Kind) -> dict[str, Any]:
    """gurobipy's handle list and handle class for `kind`, as the native calls take them."""
    if kind == "var":
        return {"pool": model._internal_vars, "cls": gp.Var}
    return {"pool": model._constrs, "cls": gp.Constr}


def added(model: Any, kind: Kind, count: int) -> None:
    """Count `count` new elements, whose handles are registered, as live in gurobipy."""
    if kind == "var":
        model._livevars += count
    else:
        model._liveconstrs += count


def register(model: Any, kind: Kind, start: int, count: int) -> None:
    """Add handles for `count` new elements from index `start` to gurobipy's bookkeeping."""
    handles = target(model, kind)
    _native.register(handles["pool"], handles["cls"], model._cmodel, start, count)
    added(model, kind, count)
