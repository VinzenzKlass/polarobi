"""
Build gurobipy models from polars frames, with data passed from Arrow memory to Gurobi's C API.

https://vinzenzklass.github.io/polarobi/
"""

from polarobi import _library
from polarobi._api import IntoColumn, Sense, VType, add_var

_library.bind()

from polarobi._namespace import GurobiFrame  # noqa: E402 - importing registers `df.grb`

__all__ = ["GurobiFrame", "IntoColumn", "Sense", "VType", "add_var"]
