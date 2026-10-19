---
title: polarobi
description: API reference of the functions and types exported by the polarobi module.
---

# `polarobi`

Functions and types importable from `polarobi` directly. Like the [`DataFrame.grb`](namespace.md) methods, the
functions take the gurobipy model to work on as their first argument and identify variables and constraints by an
`idx` column. `add_constrs` also accepts a list of term sources, for constraints that span several variable families.

::: polarobi._api
    options:
        members:
            - add_var
            - add_constrs
            - Coef
            - IntoColumn
            - Sense
            - Source
            - VType
        show_root_heading: false
        show_root_toc_entry: false
