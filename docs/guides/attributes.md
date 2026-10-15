---
title: Attributes & Results
description: Reading solutions and setting Gurobi attributes from polars frames with polarobi.
---

Every Gurobi attribute of variables and constraints, from solution values to bounds and branching priorities, can be
read and written for a whole frame at once. Both calls take the frame's `idx` column to find the elements, so they
work on any frame returned by `add_vars` or `add_constrs`, and on any filtered or joined frame derived from one.


## Reading Attributes

`get` reads an attribute for every row and returns the frame with the values as a new column named after the
attribute:

```python
model.optimize()

solution = grb(x).get(model, "X")
chosen = solution.filter(pl.col("X") > 0.5)
```

The same works for constraints and any other attribute Gurobi offers for the element, such as `"Slack"`, `"Pi"` or
`"RC"` after an LP solve, or `"LB"`, `"UB"` and `"Obj"` at any time:

```python
duals = grb(capacity_rows).get(model, "Pi")
bounds = grb(x).get(model, "UB")
```

Values come back as `Float64`. Bounds at Gurobi's infinity read as `inf`, as they do in gurobipy. Integer attributes
need `dtype=pl.Int32`:

```python
priorities = grb(x).get(model, "BranchPriority", dtype=pl.Int32)
```

Results stay in polars from here on: join them to your data, aggregate them or write them out, without a Python
object per variable.

```python
shipped = (
    grb(x).get(model, "X")
    .filter(pl.col("X") > 0.5)
    .join(orders, on="order")
    .group_by("factory", maintain_order=True)
    .agg(pl.sum("quantity"))
)
```

??? warning "Elements awaiting `update()`"
    Elements added since the last `model.update()` or `model.optimize()` have no attributes in Gurobi yet. Reading
    one raises a `GurobiError`, just as `Model.getAttr` does. Call `model.update()` first.


## Setting Attributes

`set` writes an attribute for every row. The value can be a scalar, a column name or a Series:

```python
# Pin unattainable groups to zero
grb(x.filter(pl.col("attainable").not_())).set(model, "UB", 0.0)

# Per-variable values from a column
grb(x).set(model, "Obj", "cost")

# Branch on the grouping decisions first
grb(y).set(model, "BranchPriority", 10)
```

Integer values, whether scalar or an integer column, are written through Gurobi's integer attribute API, so
attributes such as `BranchPriority` work without conversion. Everything else is written as `Float64`.

Like any attribute change in gurobipy, the new values take effect on the next `update()` or `optimize()`.


## Indices After Removals

`idx` holds the index an element had when it was added. Removing elements before it shifts indices down on the next
update, as it does for gurobipy's `Var.index`, and frames from before the removal then point at the wrong elements.
If you remove elements, read the indices of the remaining ones again, for example with
[`handles`](gurobipy.md#handles) and `Var.index`.
