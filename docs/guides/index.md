---
title: Guides
description: polarobi guides
hide:
  - toc
---

# Guides

polarobi adds four kinds of operations to polars DataFrames: adding variables, adding constraints, reading and
writing attributes, and handing elements to gurobipy. Each guide covers one of them in depth.

The examples call them as `grb(df).add_vars(...)`, after `from polarobi import GurobiFrame as grb`. This is the typed
form of `df.grb.add_vars(...)`; see [Type Checking](../installation.md#type-checking) for why.

<div class="grid cards" markdown>

- [__Variables__](variables.md) for adding a family of variables from a frame of keys
- [__Constraints__](constraints.md) for building linear constraints from frames of terms
- [__Attributes & Results__](attributes.md) for reading solutions and setting bounds, priorities and more
- [__Working with gurobipy__](gurobipy.md) for mixing polarobi with plain gurobipy code
- [__Porting a gurobipy Model__](porting.md) for moving an existing `iter_rows()` model to frames

</div>
