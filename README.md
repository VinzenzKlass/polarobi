# polarobi

Build gurobipy models from polars frames. Columns go from Arrow memory into the Gurobi C API in
one call per variable or constraint family, while the model stays fully usable through gurobipy.

```python
import polarobi  # registers df.grb

x = dist.grb.add_vars(model, by=("so_p7", "vendor_code", "ex_go_date"), vtype="B", obj="cost", name="x_dist")
caps = x.grb.add_constrs(model, vendor_cap, by=("vendor_code", "ex_go_date"), sense="<=", rhs="max_quantity", coef="quantity")
model.optimize()
solution = x.grb.get(model, "X").filter(pl.col("X") > 0)
```

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
