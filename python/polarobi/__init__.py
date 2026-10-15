"""
Build gurobipy models from polars frames, with data passed from Arrow memory to Gurobi's C API.

The API is the `grb` DataFrame namespace, registered on import:

import polarobi

x = keys.grb.add_vars(model, ...)
"""

from polarobi import _library

_library.bind()

from polarobi._namespace import GurobiFrame  # noqa: E402 - importing registers `df.grb`

__all__ = ["GurobiFrame"]
