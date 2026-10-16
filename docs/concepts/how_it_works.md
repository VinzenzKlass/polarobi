---
title: How it Works
description: How polarobi passes polars data to Gurobi's C API and keeps the model usable from gurobipy.
---

!!! tip "TL;DR"
    polarobi builds the constraint matrix with polars, hands the Arrow buffers to Gurobi's C API in one call per
    family without copying them, and then registers the new elements with gurobipy, so the model stays a regular
    gurobipy model.

## Where the Time Goes in gurobipy

Building a model with gurobipy from tabular data typically creates several Python objects for every row: a tuple
from `iter_rows()`, a lookup in a `tupledict`, a `LinExpr` or a term in one, a `quicksum` over them and a name string
from an f-string. For a few thousand rows that is irrelevant. For millions it dominates the time to build the model,
often by far.

polarobi keeps the data in polars until it reaches Gurobi. Per row, it creates exactly one Python object: the
`gp.Var` or `gp.Constr` handle that gurobipy needs to know about the element.

<div style="text-align: center" markdown>
![The gurobipy route goes from Arrow memory through Python objects per row and gurobipy to Gurobi's C library; polarobi's native module passes pointers into the Arrow buffers to Gurobi directly, then registers handles with gurobipy](../img/architecture-light.svg#only-light)
![The gurobipy route goes from Arrow memory through Python objects per row and gurobipy to Gurobi's C library; polarobi's native module passes pointers into the Arrow buffers to Gurobi directly, then registers handles with gurobipy](../img/architecture-dark.svg#only-dark)
</div>
## From Frames to Gurobi

### Variables

`add_vars` resolves the bounds, objective coefficients, type and names into columns, then makes a single
`GRBaddvars` call for the whole frame. Numeric columns are passed as they are in memory.

### Constraints

`add_constrs` turns the terms and rows into the compressed sparse row (CSR) format Gurobi's bulk API takes:

1. The rows are numbered in their order, and each term is joined to its row number.
2. The terms are sorted by row number, keeping their order within a row.
3. Each row's first term is found with a binary search over the sorted row numbers.

All of that is polars. The result, three arrays of row offsets, variable indices and coefficients, plus one array
each of senses and right-hand sides, goes to a single `GRBXaddconstrs` call.

Gurobi rejects a variable that appears twice in a row, while gurobipy sums such repeats first. Before the call, the
native module checks for repeats in one pass and, only if there are any, merges them into new arrays, summing
coefficients left to right in the order of the terms frame.

### Zero Copy

Numeric columns cross from polars to the native module through the
[Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html). The native
module reads polars' own buffers, at the same addresses polars uses, and passes pointers into them to Gurobi:

- **Single-chunk columns** are not copied at all, whatever their size.
- **Multi-chunk columns** are concatenated once, as Gurobi needs one contiguous array.
- **`UInt32` indices**, the dtype of `idx` columns, are reinterpreted as C `int` in place once they are known to fit.
- **Names** are copied, as C needs NUL-terminated strings.

Gurobi copies whatever it receives into its own storage, so the data is copied exactly once on its way in, by
Gurobi.

### Reading Attributes

`get` allocates one buffer, has Gurobi write the attribute values for all requested elements into it, and hands it
to polars as an Arrow array. polars takes the buffer over without copying it.

## Registering Handles with gurobipy

gurobipy keeps its own list of the model's variables and constraints next to Gurobi's model, and many of its
functions, such as `getVars()` and `getVarByName()`, read from that list. Elements added through the C API alone are
in the model but invisible to those functions.

So after each call, polarobi creates the gurobipy handles for the new elements and registers them exactly as
gurobipy registers its own. The native module creates them, which takes about half the time of a Python loop over
gurobipy's handle constructor; the constructor itself is gurobipy's and costs what it costs in `Model.addVars`. The
handles are the reason a variable family still takes time proportional to its size, and the price of full
compatibility with gurobipy.

This relies on private parts of gurobipy:

| Attribute                                           | Use                                  |
|-----------------------------------------------------|--------------------------------------|
| `Model._cmodel`                                     | The `GRBmodel*` of the model         |
| `Model._internal_vars`, `Model._livevars`           | gurobipy's variable list and count   |
| `Model._constrs`, `Model._liveconstrs`              | gurobipy's constraint list and count |
| `Model._has_pending_deletions()`                    | Whether removals await an update     |
| `gp.Var(cmodel, index)`, `gp.Constr(cmodel, index)` | Handle constructors                  |

That is why polarobi supports a single gurobipy minor version per release, and why its test suite checks every
model it builds against gurobipy itself (see [Equivalence with gurobipy](equivalence.md)).

## Binding to Gurobi

gurobipy ships the Gurobi library and loads it on import. polarobi never loads a second copy: two copies in one
process would hold separate state, and the `GRBmodel*` owned by gurobipy would be foreign to the second one. On
import, the native module binds to the library already loaded into the process, and checks that it reports the same
version as gurobipy.

## Performance

A reproducible benchmark builds the same allocation model with polarobi and in two common gurobipy styles, from
1,000 to ten million rows. At a million rows polarobi took 1.6 s against 9.7 s for the faster gurobipy build, and at
ten million 22.4 s against 113.4 s, with identical fingerprints; see [Benchmarks](../benchmarks.md) for the figures,
the model and how to run it yourself.
