"""
Build gurobipy models from polars frames, with data passed from Arrow memory to Gurobi's C API.

https://vinzenzklass.github.io/polarobi/
"""

from polarobi import _library

_library.bind()

from polarobi._namespace import GurobiFrame  # noqa: E402 - importing registers `df.grb`

__all__ = ["GurobiFrame"]
