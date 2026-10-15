---
title: Home
description: Build gurobipy models from polars frames, passing Arrow memory straight to Gurobi's C API.
hide:
  - navigation
  - toc
---

<h1 align="center" style="font-size: 3rem; font-weight: 600; margin: 10px 0 -15px 0; color: var(--md-typeset-color)">
polarobi
</h1>

<h1 align="center" style="font-size: 1.2rem; font-weight: 300; margin: 15px 0">
    Build gurobipy models from polars frames.
</h1>

---

polarobi adds a `grb` namespace to polars DataFrames. It turns frames of keys into Gurobi variables and frames of
terms into linear constraints, passing the data from Arrow memory straight to Gurobi's C API, one call per family.
The result is a regular gurobipy model: every variable and constraint is a real `gp.Var` / `gp.Constr`, so the rest
of your code keeps using gurobipy as before.

```python
import gurobipy as gp
import polars as pl
import polarobi  # registers df.grb

model = gp.Model()
x = orders.grb.add_vars(model, by=["order", "factory"], vtype="B", obj="value", name="x")
x.grb.add_constrs(model, capacity, by="factory", sense="<=", rhs="capacity", coef="quantity", name="cap")
model.optimize()
chosen = x.grb.get(model, "X").filter(pl.col("X") > 0.5)
```

## Key Features

- **Polars all the way**: Express variables and constraints as frames and joins instead of tuple keys, dict lookups
  and `quicksum` loops.
- **No per-row Python objects**: Numeric columns go to Gurobi straight from Arrow memory, without copying. Building
  constraints takes a fraction of the time of the equivalent gurobipy loops.
- **Fully gurobipy compatible**: Every element is a registered gurobipy handle. `getVars`, `getVarByName`, mixed
  expressions, `remove` and `copy` all work, before and after `update()`.
- **Identical models**: polarobi builds exactly the model gurobipy builds, down to names, signed zeros and
  `Fingerprint`, checked by a property-based [equivalence harness](concepts/equivalence.md).
- **Deterministic**: Constraint and term order follow your frames, so equal inputs build identical models.

---

## Getting Started

Head over to the [Installation](installation.md) and the [Quick Start](quickstart.md) for a first model, or to
[Porting a gurobipy Model](guides/porting.md) if you have one already.
