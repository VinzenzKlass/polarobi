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

!!! warning "Not on PyPI yet"
    polarobi is not published yet. Until it is, build it [from source](#building-from-source).


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
untyped. polarobi itself is fully typed, and its own sources check cleanly under strict basedpyright.
