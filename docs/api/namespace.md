---
title: DataFrame.grb
description: API reference of the grb DataFrame namespace.
---

# `DataFrame.grb`

The namespace is registered on every polars DataFrame when polarobi is imported. All methods take the gurobipy model
to work on as their first argument.

Variables and constraints are identified by an `idx` column holding their Gurobi column or row index at creation.
Like `Var.index`, these shift when elements before them are removed.

::: polarobi._namespace.GurobiFrame
    options:
        show_root_heading: false
        show_root_toc_entry: false
        filters: ["!^_"]
