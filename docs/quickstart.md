---
title: Quick Start
description: Build and solve a first Gurobi model from polars frames with polarobi.
---

This Quick Start builds a small allocation model: each order can be produced by one of several factories, each
factory has a capacity, and we pick the assignment with the highest total margin. It covers the three things you do
with polarobi: add variables, add constraints and read results.

??? info "Prerequisites"
    The Quick Start assumes you have installed polarobi and that gurobipy finds a Gurobi license. The size-limited
    license that comes with `gurobipy` is enough for this example.

```python
import gurobipy as gp
import polars as pl
import polarobi  # registers df.grb

orders = pl.DataFrame(
    {
        "order": ["o1", "o1", "o2", "o2", "o3"],
        "factory": ["f1", "f2", "f1", "f2", "f2"],
        "quantity": [40, 40, 25, 25, 60],
        "margin": [3.0, 2.5, 4.0, 3.0, 2.0],
    }
)
capacity = pl.DataFrame({"factory": ["f1", "f2"], "capacity": [50, 80]})

model = gp.Model()
```

Importing `polarobi` is what registers the `grb` namespace on every polars DataFrame, so you need it once per
process even if you never use the name.


## Adding Variables

Each row of `orders` is a candidate assignment, so we add one binary variable per row. Its objective coefficient is
the margin it earns, computed by a polars expression.

```python
x = orders.grb.add_vars(
    model,
    by=["order", "factory"],
    vtype="B",
    obj=pl.col("quantity") * pl.col("margin"),
    name="x",
)
```

??? info "Output"
    `add_vars` returns the frame with a new `idx` column: the Gurobi column index of each row's variable. This
    column is how constraints refer to variables later on.
    ```
    ┌───────┬─────────┬──────────┬────────┬─────┐
    │ order ┆ factory ┆ quantity ┆ margin ┆ idx │
    │ ---   ┆ ---     ┆ ---      ┆ ---    ┆ --- │
    │ str   ┆ str     ┆ i64      ┆ f64    ┆ u32 │
    ╞═══════╪═════════╪══════════╪════════╪═════╡
    │ o1    ┆ f1      ┆ 40       ┆ 3.0    ┆ 0   │
    │ o1    ┆ f2      ┆ 40       ┆ 2.5    ┆ 1   │
    │ o2    ┆ f1      ┆ 25       ┆ 4.0    ┆ 2   │
    │ o2    ┆ f2      ┆ 25       ┆ 3.0    ┆ 3   │
    │ o3    ┆ f2      ┆ 60       ┆ 2.0    ┆ 4   │
    └───────┴─────────┴──────────┴────────┴─────┘
    ```

The variables are named from the key columns in `by`, exactly as gurobipy's `Model.addVars` would name them:
`x[o1,f1]`, `x[o1,f2]` and so on.


## Adding Constraints

Constraints are built from a frame of *terms*: one row per variable in a constraint, with the keys that group terms
into constraints, the variable's `idx` and a coefficient. `x` is already such a frame.

First, every order is assigned at most once. Grouping `x` by `order` gives one constraint per order, with a
coefficient of 1 for each of its variables:

```python
assign = x.grb.add_constrs(model, by="order", sense="<=", rhs=1.0, coef=1.0, name="assign")
```

Second, no factory exceeds its capacity. Here the right-hand side differs per constraint, so we pass `capacity` as
the *rows* frame, one row per constraint, and take `rhs` from its column. The coefficient is each order's quantity:

```python
cap = x.grb.add_constrs(
    model, capacity, by="factory", sense="<=", rhs="capacity", coef="quantity", name="cap"
)
```

??? info "Output"
    Both calls return their rows with the constraints' row index in `idx`.
    ```
    ┌───────┬─────┐        ┌─────────┬──────────┬─────┐
    │ order ┆ idx │        │ factory ┆ capacity ┆ idx │
    │ ---   ┆ --- │        │ ---     ┆ ---      ┆ --- │
    │ str   ┆ u32 │        │ str     ┆ i64      ┆ u32 │
    ╞═══════╪═════╡        ╞═════════╪══════════╪═════╡
    │ o1    ┆ 0   │        │ f1      ┆ 50       ┆ 3   │
    │ o2    ┆ 1   │        │ f2      ┆ 80       ┆ 4   │
    │ o3    ┆ 2   │        └─────────┴──────────┴─────┘
    └───────┴─────┘
    ```

??? info "The resulting model"
    Written by `model.write("quickstart.lp")`:
    ```
    Maximize
      120 x[o1,f1] + 100 x[o1,f2] + 100 x[o2,f1] + 75 x[o2,f2] + 120 x[o3,f2]
    Subject To
     assign[o1]: x[o1,f1] + x[o1,f2] <= 1
     assign[o2]: x[o2,f1] + x[o2,f2] <= 1
     assign[o3]: x[o3,f2] <= 1
     cap[f1]: 40 x[o1,f1] + 25 x[o2,f1] <= 50
     cap[f2]: 40 x[o1,f2] + 25 x[o2,f2] + 60 x[o3,f2] <= 80
    Binaries
     x[o1,f1] x[o1,f2] x[o2,f1] x[o2,f2] x[o3,f2]
    End
    ```


## Solving and Reading Results

The model is a plain gurobipy model, so solving it is too:

```python
model.ModelSense = gp.GRB.MAXIMIZE
model.optimize()
```

`get` reads an attribute for every row of a frame and adds it as a column, so the solution comes back as a frame
you can filter and join like any other:

```python
chosen = x.grb.get(model, "X").filter(pl.col("X") > 0.5)
slack = cap.grb.get(model, "Slack")
```

??? info "Output"
    ```
    ┌───────┬─────────┬──────────┬────────┬─────┬─────┐
    │ order ┆ factory ┆ quantity ┆ margin ┆ idx ┆ X   │
    │ ---   ┆ ---     ┆ ---      ┆ ---    ┆ --- ┆ --- │
    │ str   ┆ str     ┆ i64      ┆ f64    ┆ u32 ┆ f64 │
    ╞═══════╪═════════╪══════════╪════════╪═════╪═════╡
    │ o1    ┆ f1      ┆ 40       ┆ 3.0    ┆ 0   ┆ 1.0 │
    │ o3    ┆ f2      ┆ 60       ┆ 2.0    ┆ 4   ┆ 1.0 │
    └───────┴─────────┴──────────┴────────┴─────┴─────┘
    ┌─────────┬──────────┬─────┬───────┐
    │ factory ┆ capacity ┆ idx ┆ Slack │
    │ ---     ┆ ---      ┆ --- ┆ ---   │
    │ str     ┆ i64      ┆ u32 ┆ f64   │
    ╞═════════╪══════════╪═════╪═══════╡
    │ f1      ┆ 50       ┆ 3   ┆ 10.0  │
    │ f2      ┆ 80       ┆ 4   ┆ 20.0  │
    └─────────┴──────────┴─────┴───────┘
    ```


## Next Steps

The [Guides](guides/index.md) cover each part in depth, starting with [Variables](guides/variables.md) and
[Constraints](guides/constraints.md). If you already have a gurobipy model built with loops over `iter_rows()`, see
[Porting a gurobipy Model](guides/porting.md).
