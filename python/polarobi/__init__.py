"""
Build gurobipy models from polars frames, with data passed from Arrow memory to Gurobi's C API.

https://vinzenzklass.github.io/polarobi/
"""

from polarobi import _library
from polarobi._api import Coef, IntoColumn, Sense, Source, VType, add_constr, add_constrs, add_var

_library.bind()

from polarobi._namespace import GurobiFrame  # noqa: E402 - importing registers `df.grb`

__all__ = [
    "Coef",
    "GurobiFrame",
    "IntoColumn",
    "Sense",
    "Source",
    "VType",
    "add_constr",
    "add_constrs",
    "add_var",
]
