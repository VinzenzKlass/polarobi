# polarobi

Build gurobipy models from polars frames. Columns go from Arrow memory into the Gurobi C API in
one call per variable or constraint family, while the model stays fully usable through gurobipy.

```python
import gurobipy as gp
import polars as pl
from polarobi import GurobiFrame as grb

orders = pl.DataFrame(
    {
        "order": ["o1", "o1", "o2", "o2", "o3"],
        "factory": ["f1", "f2", "f1", "f2", "f2"],
        "quantity": [40, 40, 25, 25, 60],
        "value": [120.0, 100.0, 100.0, 75.0, 120.0],
    }
)
capacity = pl.DataFrame({"factory": ["f1", "f2"], "capacity": [50, 80]})

model = gp.Model()
model.ModelSense = gp.GRB.MAXIMIZE

# One binary variable per order and factory, worth the order's value.
x = grb(orders).add_vars(model, by=["order", "factory"], vtype="B", obj="value", name="x")
# Each order is made at most once, and no factory exceeds its capacity.
grb(x).add_constrs(model, by="order", sense="<=", rhs=1.0, coef=1.0, name="assign")
grb(x).add_constrs(
    model, capacity, by="factory", sense="<=", rhs="capacity", coef="quantity", name="cap"
)

model.optimize()
print(grb(x).get(model, "X").filter(pl.col("X") > 0.5))
```

`grb(df)` is the typed form of polars' `df.grb` namespace: both do the same, but type checkers only see the first.
See [Type Checking](docs/installation.md#type-checking) for why.

## How it works

- `Model._cmodel` holds the `GRBmodel*`. The native module binds to the Gurobi library that
  gurobipy already loaded (never a second copy) and checks that both report the same version.
- Numeric columns are passed without copying (one concatenation if a column has several chunks).
  Names are copied, since C needs NUL-terminated strings.
- New columns and rows are registered with gurobipy as `gp.Var(cmodel, i)` / `gp.Constr(cmodel, i)`
  handles in `Model._internal_vars` / `Model._constrs`. That is what keeps `getVars`,
  `getVarByName`, mixed expressions, `remove`, `copy` and friends working. Creating the handles
  costs about 2 µs each, the same as in `addVars`.
- Constraint and term order follow the input frames, so equal inputs build models with equal
  `Fingerprint`s.

This relies on private gurobipy attributes (`_cmodel`, `_internal_vars`, `_livevars`, `_constrs`,
`_liveconstrs`, `_has_pending_deletions`). Pin gurobipy to a minor version and keep
`tests/test_compat.py` in CI.

## Layout

- `src/`: the Rust extension (`pyo3`, `pyo3-arrow`), built as module `polarobi._native`.
- `python/polarobi/`: the Python package with the `df.grb` namespace.

Build with `uv sync` (maturin backend, see `pyproject.toml`); run the tests with `uv run pytest`.
