---
title: Constraints
description: Building linear constraints from polars frames of terms with polarobi.
---

`add_constrs` adds a family of linear constraints in one call. Instead of an expression per constraint, you describe
the whole family with two frames:

- **Terms**, the frame you call `add_constrs` on: one row per term, that is per variable in a constraint. It holds
  the key columns that say which constraint a term belongs to, the variable's index and its coefficient.
- **Rows**, an optional second frame: one row per constraint, holding its keys and anything that varies per
  constraint, such as the right-hand side.

Each group of terms with the same key `k` becomes one constraint:

$$
\sum_{t \in T_k} \text{coef}_t \, x_{\text{var}_t} \;\lessgtr\; \text{rhs}_k
$$

where $T_k$ are the terms with key $k$ and $\lessgtr$ is the constraint's sense.

Because both are ordinary frames, you build them with ordinary polars: joins, filters and `concat` take the place
of tuple lookups and `quicksum` loops.


## Basic Usage

The simplest case groups a variable frame by one of its columns. Here, every order is shipped at most once:

$$
\sum_{w} x_{o,w} \le 1 \qquad \forall o
$$

```python
x = grb(shipments).add_vars(model, by=["order", "week"], vtype="B", name="ship")
once = grb(x).add_constrs(model, by="order", sense="<=", rhs=1.0, name="once")
```

`x` already has an `idx` column, which is where `add_constrs` looks for the variable index by default, so it can be
used as the terms frame directly, and every term gets the default coefficient of 1.

`add_constrs` returns the rows, here the distinct orders, with each constraint's row index in an `idx` column.


## Terms

The terms frame needs the key columns `by` and a variable index column, which is `idx` by default; pass
`var="..."` if it has another name. Coefficients come from the `coef` argument, which, like the
[bounds of variables](variables.md#bounds-and-objective), accepts a scalar, a column name, an expression or a Series.
It defaults to 1, as in gurobipy's `quicksum`, so weighted terms always name their coefficient, as in `coef="coef"`
below. A frame's `coef` column is not used unless you pass it.

Terms usually come from joining a variable frame with your data. To limit the tooling each product needs per week,
join the shipments to the products' tooling:

$$
\sum_{p \,\in\, P_k} \text{hours}_{k,p} \, x_{p,w} \le \text{capacity}_{k,w} \qquad \forall (k, w)
$$

where $P_k$ are the products that use tooling $k$.

```python
terms = x.join(tooling, on="product").select("tooling", "week", "idx", coef=pl.col("hours"))
grb(terms).add_constrs(
    model, tooling_capacity, by=["tooling", "week"], sense="<=", rhs="capacity", coef="coef"
)
```

A constraint that spans several variable families takes a list of term sources with the module-level `add_constrs`.
Each source is `frame` (coefficient 1, index `idx`), `(frame, coef)` or `(frame, coef, var)`. The capacity of a
factory, used by both kinds of demand:

$$
\sum_{d} \text{quantity}_d \, x^{\text{dist}}_{d} + \sum_{s} \text{quantity}_s \, x^{\text{stock}}_{s} \le \text{max\_quantity}_{f,w} \qquad \forall (f, w)
$$

with $d$ and $s$ ranging over the terms of factory $f$ in week $w$.

```python
from polarobi import add_constrs

terms = [(x_dist, "quantity"), (x_stock, "quantity")]
add_constrs(model, terms, factories, by=["factory", "week"], sense="<=", rhs="max_quantity")
```

Variables that appear on the right-hand side of a mathematical formulation move to the left with a negated
coefficient. A minimum order quantity with a binary $y_g$ per group,

$$
\sum_{i \in g} q_i \, x_i \ge \text{moq}_g \, y_g
\quad\Longrightarrow\quad
\sum_{i \in g} q_i \, x_i - \text{moq}_g \, y_g \ge 0 \qquad \forall g,
$$

which in polarobi is:

```python
add_constrs(
    model, [(x, "quantity"), (y, -pl.col("moq"))], by="group", sense=">=", rhs=0.0, name="moq"
)
```

Sources need the `by` columns and may carry any others. Key dtypes are unified across sources (an `Int32` and an
`Int64` key become `Int64`); keys that can't be unified raise a `TypeError` naming the source. Terms keep source
order, and a variable repeated in one constraint has its coefficients summed. `add_constrs(model, [(df, c, v)], ...)`
is the same as `grb(df).add_constrs(model, ..., coef=c, var=v)`.

!!! tip "Index dtypes"
    `idx` columns from `add_vars` are `UInt32`, and are passed to Gurobi without copying. Other integer dtypes work
    too and are converted once.


## Rows

Without a rows frame, there is one constraint per distinct key in the terms, in order of first appearance, and
`sense`, `rhs` and `name` can only be the same for all of them. Pass a rows frame to vary them per constraint, or to
control which constraints exist and in which order:

$$
\sum_{i \in f} \text{quantity}_i \, x_i \le \text{capacity}_f \qquad \forall f \in \text{factories}
$$

```python
factories = pl.DataFrame({"factory": ["f1", "f2"], "capacity": [50.0, 80.0]})
grb(x).add_constrs(model, factories, by="factory", sense="<=", rhs="capacity", coef="quantity")
```

Terms whose keys have no row are not part of any constraint. Rows whose keys have no terms are dropped, as an empty
constraint is usually a mistake in the data rather than intent; pass `drop_empty=False` to add them anyway, as
constraints of the form `0 <= rhs`.

The returned frame is the rows frame, after dropping empty rows, with the constraints' `idx`. Its other columns are
kept, so you can join results back to your data later.

Rows may repeat a key. Every such row becomes a constraint of its own over all terms with that key, as with
gurobipy's `Model.addConstrs`.


## Sense and Right-Hand Side

`sense` is one of `"<"`, `"<="`, `">"`, `">="`, `"="` or `"=="`, or the `gp.GRB` constants, for the whole family.
To mix senses, pass an expression or a Series of these strings:

```python
rows = pl.DataFrame({"factory": ["f1", "f2"], "bound": [10.0, 80.0], "kind": [">=", "<="]})
grb(x).add_constrs(model, rows, by="factory", sense=pl.col("kind"), rhs="bound", coef="quantity")
```

`rhs` takes the same forms as `coef`: a scalar, a column name, an expression or a Series. Expressions and column
names are evaluated against the rows frame, and a Series lines up with the rows frame as you pass it, before empty
rows are dropped.


## A Single Constraint

`add_constr` sums every term into one constraint and returns a one-row frame with its `idx`. A global limit on air
freight, for example:

$$
\sum_{i \,\in\, \text{AIR}} \text{quantity}_i \, x_i \le \text{max\_air}
$$

```python
air = x.filter(pl.col("mode") == "AIR")
grb(air).add_constr(model, sense="<=", rhs=max_air, coef="quantity", name="air_limit")
```

Like `add_constrs`, the module-level `polarobi.add_constr` also accepts a list of term sources, and `by=[]` in
`add_constrs` does the same. `name` is the constraint's plain name, without brackets.


## Repeated Variables

A variable may appear more than once in the terms of a constraint. Its coefficients are then summed, left to right
in the order of the terms frame, and Gurobi drops results that are zero or so small that they are numerically zero,
as it does for gurobipy's expressions.

gurobipy sums repeats too, but in the order of an unstable sort that depends on the row's other terms. Where a sum
depends on its order, the last bit of the coefficient can therefore differ from a row built with gurobipy.
polarobi's order is fixed and the same on every platform.


## Names

A `name` string names constraints from their keys, as `Model.addConstrs` does: `cap[f1]`, `tooling[t1,2026-01]`.
Keys are formatted as for [variables](variables.md#names). An expression is evaluated against the rows frame and
names each constraint freely.


## Order and Determinism

Constraints are added in the order of the rows frame, or of first appearance in the terms if there is none, and
terms keep their order within a constraint. Equal frames therefore always build identical models, with the same
`Fingerprint`.

Polars does not keep the order of `group_by` or `unique` unless asked to. If your terms or rows come out of either,
pass `maintain_order=True`, or sort them, to keep the model the same from run to run.
