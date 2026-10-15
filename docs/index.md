---
title: Home
description: Build gurobipy models from polars frames, passing Arrow memory straight to Gurobi's C API.
hide:
  - navigation
  - toc
---

<p align="center" style="margin: -15px 0 10px 0">
    <img width="320"  src="img/logo.webp" alt='' style="border-radius: 15px">
</p>

<h1 align="center" style="font-size: 3rem; font-weight: 600; margin: 10px 0 -15px 0; color: var(--md-typeset-color)">
polarobi
</h1>

<h1 align="center" style="font-size: 1.2rem; font-weight: 300; margin: 15px 0">
   Build gurobipy models from polars frames, passing Arrow memory straight to Gurobi's C API.
</h1>

---

polarobi adds a `grb` namespace to polars DataFrames. It turns frames of keys into Gurobi variables and frames of
terms into linear constraints, passing the data from Arrow memory straight to Gurobi's C API, one call per family.
The result is a regular gurobipy model: every variable and constraint is a real `gp.Var` / `gp.Constr`, so the rest
of your code keeps using gurobipy as before.

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
grb(x).add_constrs(model, by="order", sense="<=", rhs=1.0, name="assign")
grb(x).add_constrs(
    model, capacity, by="factory", sense="<=", rhs="capacity", coef="quantity", name="cap"
)

model.optimize()
print(grb(x).get(model, "X").filter(pl.col("X") > 0.5))
```

`grb(df)` is the typed form of polars' `df.grb` namespace: both do the same, but type checkers only see the first.
See [Type Checking](installation.md#type-checking) for why.

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
