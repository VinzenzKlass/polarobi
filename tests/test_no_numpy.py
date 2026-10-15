"""numpy is a dev dependency for type checking only, polarobi must never import it at runtime."""

import subprocess
import sys

_SCRIPT = """
import sys
import gurobipy as gp
import polars as pl
import polarobi

with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as model:
    x = pl.DataFrame({"k": ["a", "b"], "q": [1.0, 2.0]}).grb.add_vars(model, by=["k"], obj="q")
    terms = x.with_columns(g=pl.lit("all"), coef=pl.lit(1.0))
    terms.grb.add_constrs(model, by="g", sense="<", rhs=1.0)
    model.optimize()
    x.grb.get(model, "X")
    x.grb.set(model, "BranchPriority", 1)
    x.grb.get(model, "BranchPriority", dtype=pl.Int32)
assert "numpy" not in sys.modules, "polarobi imported numpy"
"""


def test_polarobi_never_imports_numpy() -> None:
    subprocess.run([sys.executable, "-c", _SCRIPT], check=True)
