---
title: Variables
description: Adding Gurobi variables from polars frames with polarobi.
---

`add_vars` adds one variable per row of a frame. It is the frame equivalent of gurobipy's `Model.addVars`: the frame's
rows play the role of the keys, and its columns supply bounds, objective coefficients and names.

## Basic Usage

```python
import gurobipy as gp
import polars as pl
from polarobi import GurobiFrame as grb

model = gp.Model()
shipments = pl.DataFrame(
    {
        "order": ["o1", "o1", "o2"],
        "week": ["2026-01", "2026-02", "2026-01"],
        "cost": [12.0, 9.5, 7.0],
    }
)

x = grb(shipments).add_vars(model, by=["order", "week"], ub=1.0, obj="cost", name="ship")
```

`add_vars` returns the input frame with one more column, `idx`, holding each variable's Gurobi column index. Keep
this frame: it is how you refer to the variables when you add [constraints](constraints.md) or read
[results](attributes.md). The column name is configurable with `idx="..."`, which is useful when one frame holds
several families.

Indices are positions in the whole model, not in the family. A second `add_vars` call continues where the first one
ended, after any variables gurobipy added in between.


## Bounds and Objective

`lb`, `ub` and `obj` each accept any of these forms:

| Form | Example | Meaning |
|---|---|---|
| Omitted | | Gurobi's default: `lb=0`, `ub=inf`, `obj=0` |
| Scalar | `ub=1.0` | The same value for every variable |
| Column name | `obj="cost"` | The column's value for each row |
| Expression | `obj=pl.col("cost") * 1.1` | Evaluated against the frame |
| Literal expression | `ub=pl.lit(10.0)` | The same value for every variable |
| Series | `lb=pl.Series([0.0, 1.0, 0.0])` | One value per row, in row order |

Values are cast to `Float64`. Infinite bounds are written as `float("inf")`, or `gp.GRB.INFINITY` as in gurobipy.
Nulls are rejected: Gurobi has no notion of a missing bound.


## Variable Types

`vtype` takes one of gurobipy's type codes for the whole family: `"C"` (continuous, the default), `"B"` (binary),
`"I"` (integer), `"S"` (semi-continuous) or `"N"` (semi-integer). The `gp.GRB` constants work as well:

```python
y = grb(sites).add_vars(model, vtype=gp.GRB.BINARY, name="open")
```


## Names

A `name` string names variables the way `Model.addVars` does: the name, then the values of the key columns in
brackets. The key columns are the ones listed in `by`, or all of the frame's columns if `by` is omitted.

```python
x = grb(shipments).add_vars(model, by=["order", "week"], name="ship")
# ship[o1,2026-01], ship[o1,2026-02], ship[o2,2026-01]
```

Keys are formatted exactly as gurobipy formats them, which is Python's `str()` of each value:

| Key type | Value | In the name |
|---|---|---|
| String | `"o1"` | `o1` |
| Integer | `7` | `7` |
| Boolean | `True` | `True` |
| Float | `1.5e-07` | `1.5e-07` |
| Date | `2026-01-02` | `2026-01-02` |
| Null | | `None` |

Other key types, such as datetimes, raise a `TypeError` when used for names. For anything else, pass an
expression; it is evaluated per row and may produce any string:

```python
x = grb(shipments).add_vars(model, name=pl.concat_str(pl.lit("ship_"), pl.col("order")))
```

A null produced by a name expression leaves that variable unnamed, so Gurobi gives it its default name, as an empty
name in gurobipy would.

!!! tip "Names are optional"
    Without `name`, variables get Gurobi's default names (`C0`, `C1`, ...). Names are copied to C as strings, so
    for very large models, leaving them out saves a little time and memory.

??? warning "`pl.format` in polars 2.0.0"
    In polars 2.0.0, `pl.format(...)` can panic on a string column that consists of several chunks and was filtered
    before. `pl.concat_str` builds the same names and is not affected.


## Duplicate Keys

Like `Model.addVars`, `add_vars` raises a `KeyError` if two rows have the same values in the key columns `by`.
Unlike gurobipy, which adds the variables before it raises, polarobi checks first and leaves the model unchanged.

Pass `by=[]` to add variables without keys. They are then never duplicates, and a `name` string becomes every
variable's plain name.


## Large Frames

`add_vars` passes numeric columns to Gurobi without copying them and makes a single call into Gurobi for the whole
frame. What remains per variable is creating its gurobipy handle, which `Model.addVars` pays as well.
[How it Works](../concepts/how_it_works.md) has the details.
