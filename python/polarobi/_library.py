import sys
from pathlib import Path

import gurobipy as gp

from polarobi import _native


def _loaded_library() -> str:
    if sys.platform == "linux":
        # Whatever is actually mapped, regardless of how the wheel names or places it.
        with open("/proc/self/maps") as maps:
            for line in maps:
                parts = line.split(maxsplit=5)
                if len(parts) == 6 and "libgurobi" in Path(parts[5].strip()).name:
                    return parts[5].strip()
        raise ImportError("polarobi: no Gurobi library is mapped into this process.")
    pattern = "gurobi*.dll" if sys.platform == "win32" else "**/libgurobi*.dylib"
    root = Path(gp.__file__).parent
    candidates = sorted(root.glob(pattern)) or sorted(root.parent.glob(f"gurobipy*/{pattern}"))
    if not candidates:
        raise ImportError(f"polarobi: no Gurobi library found next to gurobipy in {root}.")
    return str(candidates[0])


def bind() -> None:
    gp.Model  # noqa: B018 - the library is loaded by gurobipy's extension modules on import
    native = _native.load(_loaded_library())
    expected = gp.gurobi.version()
    if native != expected:
        raise ImportError(
            f"polarobi: bound Gurobi {native} but gurobipy runs {expected}; "
            "the native module must share gurobipy's library."
        )
