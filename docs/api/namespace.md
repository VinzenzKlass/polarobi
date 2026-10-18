---
title: DataFrame.grb
description: API reference of the grb DataFrame namespace.
---

# `DataFrame.grb`

The namespace is registered on every polars DataFrame when polarobi is imported. It is the class
`polarobi.GurobiFrame`, so `GurobiFrame(df)`, or `grb(df)` with `from polarobi import GurobiFrame as grb`, is the
same namespace in a form type checkers understand; see [Type Checking](../installation.md#type-checking). All methods
take the gurobipy model to work on as their first argument.

Variables and constraints are identified by an `idx` column holding their Gurobi column or row index at creation.
Like `Var.index`, these shift when elements before them are removed.

::: polarobi._api
    options:
        members:
            - IntoColumn
            - Sense
            - VType
        show_root_heading: false
        show_root_toc_entry: false

::: polarobi._namespace.GurobiFrame
    options:
        show_root_heading: false
        show_root_toc_entry: false
        filters: ["!^_"]
