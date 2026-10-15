---
title: Installation
description: How to install polarobi.
---

polarobi requires Python 3.12 or higher.

=== "pip"
    ```shell
    pip install polarobi
    ```
===+ "uv"
    ```shell
    uv add polarobi
    ```
=== "Poetry"
    ```shell
    poetry add polarobi
    ```
### Dependencies

polarobi has just two dependencies:

- [gurobipy](https://pypi.org/project/gurobipy/): The Gurobi Python API, and with it the Gurobi library itself.
- [polars](https://pola.rs/): DataFrames.

It does not need numpy. polars exchanges data with the native module through the
[Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html).

???+ note "gurobipy is pinned to a minor version"
    polarobi registers its variables and constraints with gurobipy through some of gurobipy's private attributes,
    so each polarobi release supports a single gurobipy minor version, currently 14.0. See
    [How it Works](concepts/how_it_works.md#registering-handles-with-gurobipy) for the details.


### Gurobi License

polarobi uses whatever license gurobipy finds. The size-limited license that comes with the `gurobipy` package is
enough to try it on small models; anything larger needs a full Gurobi license.


### Building from Source

The native module is written in Rust and built with [maturin](https://www.maturin.rs/). You need a
[Rust toolchain](https://rustup.rs/) and [uv](https://docs.astral.sh/uv/):

```shell
git clone <repository> && cd polarobi
uv sync
```

`uv sync` compiles the extension and installs polarobi into the project's environment, rebuilding it whenever the
Rust sources change. `uv build` produces a wheel for your platform.


### Type Checking

polars registers namespaces at runtime, so type checkers do not know about `df.grb` and treat its calls as
untyped. Python's type system has no way to declare an attribute on another package's class, so there is no
annotation that fixes that.

`df.grb` is an instance of `polarobi.GurobiFrame`, constructed with the frame. Constructing it yourself gives you
the same namespace, fully typed. A short alias keeps the calls readable:

```python
import gurobipy as gp
import polars as pl
from polarobi import GurobiFrame as grb

orders = pl.DataFrame({"order": ["o1", "o2"], "value": [3.0, 5.0]})
model = gp.Model()

x = grb(orders).add_vars(model, by=["order"], vtype="B", obj="value")
assign = grb(x).add_constrs(model, by=[], sense="<=", rhs=1.0, coef=1.0)
constraints = grb(assign).handles(model, kind="constr")  # list[gp.Constr]
```

Type checkers then check every argument and know every return type: `add_vars` returns a `pl.DataFrame`,
`handles(kind="constr")` a `list[gp.Constr]`, and a misspelled keyword is an error.

In a chain of polars calls, `pipe` does the same:

```python
x = orders.filter(pl.col("value") > 4).pipe(grb).add_vars(model, vtype="B", obj="value")
```

`grb(df)` and `df.grb` are the same class doing the same work, so both forms can be mixed freely. polarobi itself
is fully typed, and its own sources check cleanly under strict basedpyright.
