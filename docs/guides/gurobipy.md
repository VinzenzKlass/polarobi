---
title: Working with gurobipy
description: Mixing polarobi and plain gurobipy code in one model.
---

polarobi does not wrap or replace gurobipy. It adds variables and constraints to a regular `gp.Model` and registers
each of them with gurobipy as an ordinary `gp.Var` or `gp.Constr`. You can therefore use polarobi for the large,
regular families and plain gurobipy for everything else, in any order, in the same model.


## Handles

`handles` returns the gurobipy objects of a frame's rows, in row order:

```python
variables = grb(x).handles(model)
constraints = grb(caps).handles(model, kind="constr")
```

They are the same objects gurobipy itself returns:

```python
model.update()
first = model.getVarByName("ship[o1,2026-01]")
assert first is not None and first.sameAs(variables[0])
```

`tupledict` returns them keyed like `Model.addVars` keys its result: by the value of a single key column, or by
tuples of several:

```python
ship = grb(x).tupledict(model, ["order", "week"])
model.addLConstr(ship["o1", "2026-01"] + ship["o1", "2026-02"] <= 1)
total = ship.sum("o1", "*")
```

Neither creates new handles: they exist from the moment polarobi adds the elements.


## Mixing Both APIs

Everything gurobipy offers works on elements polarobi added, and polarobi works on models gurobipy built:

- `getVars()`, `getConstrs()`, `getVarByName()`, `getConstrByName()` and `getRow()` see all elements.
- Expressions mix variables from both sides, before or after `update()`.
- `remove()`, `copy()`, `write()` and `optimize()` behave as usual.
- Terms frames may refer to variables gurobipy added, by their index (for a single variable without an `update()`,
  use [`add_var`](variables.md#single-variables)):

```python
from polarobi import add_var

air = add_var(model, ub=budget, name="air_expansion")

terms = pl.concat(
    [
        x.filter(pl.col("mode") == "AIR").select("idx", coef=pl.col("quantity")),
        air.with_columns(coef=pl.lit(-1.0)),
    ],
    how="vertical_relaxed",
)
grb(terms).add_constrs(model, by=[], sense="<=", rhs=max_air, coef="coef", name="air_limit")
```

!!! note "Pending elements"
    gurobipy updates models lazily: new elements have no index-based attributes until `update()` or `optimize()`.
    polarobi follows the same rules. It does not call `model.update()` unless the model has pending deletions from
    `model.remove(...)`, so it can add constraints over variables that await an update, while reading their attributes
    needs an update first, exactly as with gurobipy.


## Removing Elements

Removing variables or constraints works through gurobipy, with `model.remove(...)`. As with gurobipy's `Var.index`,
the indices of later elements shift down once the removal is applied. Whenever the model has pending deletions, the
next polarobi call runs `model.update()` first so its own calls see current indices; `idx` columns in frames from
before the removal, however, are not updated. See [Indices After Removals](attributes.md#indices-after-removals).


## Objectives and Model Settings

Objective coefficients set with `obj=` in [`add_vars`](variables.md#bounds-and-objective) are part of the model's
linear objective, just like `obj=` in gurobipy. Everything about the model as a whole, such as the objective sense,
parameters and `setObjective` for objectives that are easier to write as expressions, stays with gurobipy:

```python
model.ModelSense = gp.GRB.MAXIMIZE
model.Params.MIPGap = 0.01
```
