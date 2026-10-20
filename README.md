# polarobi

> [!WARNING]
> polarobi is unstable and experimental. Its API may change without notice, and it builds on private
> internals of `gurobipy`. Use it for production models only if you can verify the result, for example by
> comparing the model's Fingerprint with a model built another way, or if you have other strong guarantees
> that the model is correct.

Build gurobipy models from polars frames. Columns go from Arrow memory into the Gurobi C API in
one call per variable or constraint family, while the model stays fully usable through gurobipy.

```python
import gurobipy as gp
import polars as pl
from polarobi import GurobiFrame as grb

orders = pl.DataFrame(
    {
        "order": ["o1", "o1", "o2", "o2", "o3"],
        "factory": ["f1", "f2", "f1", "f2", "f2"],
        "quantity": [40, 40, 25, 25, 60],
        "value": [120.0, 100.0, 100.0, 75.0, 120.0],
    }
)
capacity = pl.DataFrame({"factory": ["f1", "f2"], "capacity": [50, 80]})

model = gp.Model()
model.ModelSense = gp.GRB.MAXIMIZE

# One binary variable per order and factory, worth the order's value.
x = grb(orders).add_vars(model, by=["order", "factory"], vtype="B", obj="value", name="x")
# Each order is made at most once, and no factory exceeds its capacity.
grb(x).add_constrs(model, by="order", sense="<=", rhs=1.0, name="assign")
grb(x).add_constrs(
    model, capacity, by="factory", sense="<=", rhs="capacity", coef="quantity", name="cap"
)

model.optimize()
print(grb(x).get(model, "X").filter(pl.col("X") > 0.5))
```

`grb(df)` is the typed form of polars' `df.grb` namespace: both do the same, but type checkers only see the first.
See [Type Checking](https://vinzenzklass.github.io/polarobi/installation/#type-checking) for why.

## How it works

- `Model._cmodel` holds the `GRBmodel*`. The native module binds to the Gurobi library that
  gurobipy already loaded (never a second copy) and checks that both report the same version.
- Numeric columns are passed without copying (one concatenation if a column has several chunks).
  Names are copied, since C needs NUL-terminated strings.
- New columns and rows are registered with gurobipy as `gp.Var(cmodel, i)` / `gp.Constr(cmodel, i)`
  handles in `Model._internal_vars` / `Model._constrs`. That is what keeps `getVars`,
  `getVarByName`, mixed expressions, `remove`, `copy` and friends working. The handles are created
  in Rust, at about half the cost of a Python loop over gurobipy's handle constructor.
- Constraint and term order follow the input frames, so equal inputs build models with equal
  `Fingerprint`s.

This relies on private gurobipy attributes (`_cmodel`, `_internal_vars`, `_livevars`, `_constrs`,
`_liveconstrs`, `_has_pending_deletions`). Pin gurobipy to a minor version and keep
`tests/test_compat.py` in CI.

## Contributing

polarobi is a mixed Rust and Python project. The Rust extension in `src/` is compiled by
[maturin](https://www.maturin.rs/) into the module `polarobi._native` and packaged together with the Python sources in
`python/polarobi/`. [uv](https://docs.astral.sh/uv/) drives the whole thing: maturin is the project's build backend,
so the usual uv commands compile the extension when needed.

### Prerequisites

- **Rust**, stable, installed with [rustup](https://rustup.rs/), with the `rustfmt` and `clippy` components for the
  pre-commit hooks:
  ```shell
  rustup component add rustfmt clippy
  ```
  On Windows, rustup's default toolchain needs the MSVC build tools (Visual Studio Build Tools, "Desktop development
  with C++"). On macOS, the Xcode command line tools.
- **uv**, which also provides Python. polarobi supports Python 3.12 and up; CI runs 3.14.
- **A Gurobi license** that gurobipy finds. The size-limited license that ships with `gurobipy` is enough for the tests.

You do not need a Gurobi installation or SDK to compile: the extension does not link against Gurobi. At import it binds
to the Gurobi library that gurobipy has already loaded into the process.

### Setting up

```shell
uv sync --all-groups
uv run prek install
```

`uv sync --all-groups` creates `.venv`, installs the dev and benchmark dependencies and builds polarobi with
maturin as an editable install: a `.pth` file points at `python/`, and the compiled extension is placed next to the
Python sources as `python/polarobi/_native.*.pyd` (or `.so`). The type check covers the benchmarks too, so it needs
the `bench` group. `prek install` sets up the pre-commit hooks below.

### The build loop

- **Python changes** take effect immediately, the package is installed in editable mode.
- **Rust changes** are picked up by the next `uv sync` or `uv run`: `[tool.uv] cache-keys` in `pyproject.toml` lists
  `Cargo.toml` and `src/**/*.rs`, so uv rebuilds the extension when they change. This builds in release mode, which
  takes a while to compile.
- **For faster iterations** on Rust code, build a debug extension in place and keep uv from rebuilding over it:
  ```shell
  uv run --no-sync maturin develop --uv
  uv run --no-sync pytest
  ```
  Debug builds compile quickly but run slowly, so use `maturin develop --uv --release`, or a plain `uv sync`, before
  benchmarking.
- **`cargo check`** and **`cargo clippy`** work on `src/` directly and are the quickest way to see compile errors.


### Building wheels

```shell
uv build
```

builds an sdist and a wheel for the current platform into `dist/`. For a manylinux wheel from another platform, maturin
can cross-compile with [zig](https://ziglang.org/):

```shell
uvx maturin build --release --zig --target x86_64-unknown-linux-gnu
```

Wheels use Python's stable ABI (pyo3's `abi3-py312` feature), so one wheel per platform, tagged `cp312-abi3`, works on
every Python from 3.12 on.

### Checks

The pre-commit hooks in `prek.toml` run on every commit. Run them on everything with:

```shell
uv run prek run --all-files
```

| Hook                        | Checks                                            |
|-----------------------------|---------------------------------------------------|
| `uv-lock`                   | `uv.lock` matches `pyproject.toml`                |
| `ruff-format`, `ruff-check` | Python formatting and lints, fixed in place       |
| `basedpyright`              | strict type checking of the package and the tests |
| `cargo fmt`                 | Rust formatting, fixed in place                   |
| `cargo clippy`              | Rust lints, with warnings as errors               |
| `cargo test`                | Rust unit tests of the parts that need no Gurobi  |

A hook that fixes files in place fails the commit; add the fixed files and commit again.

### Tests

```shell
uv run pytest
cargo test
```

`cargo test` covers what the native module does without Gurobi: reading Arrow columns and lists, the CSR checks
and merging repeated variables. Some of its tests start an embedded Python to check error messages. Everything that
calls Gurobi is tested through `pytest` against gurobipy.

Besides unit tests, `tests/test_equivalence.py` runs a [Hypothesis](https://hypothesis.readthedocs.io/) state machine
that checks every model polarobi builds against the same model built with plain gurobipy, down to the MPS file and
`Fingerprint`. The default profile is quick; set `HYPOTHESIS_PROFILE=ci` for ten times the examples, or `thorough` for a
long search after changes to the native code or the namespace semantics:

```shell
HYPOTHESIS_PROFILE=thorough uv run pytest tests/test_equivalence.py
```

`tests/harness/__init__.py` explains how to extend the harness for a new feature, and
[Equivalence with gurobipy](docs/concepts/equivalence.md) what it guarantees.

### Benchmarks

`benchmarks/build_model.py` builds the same allocation model with polarobi, in two plain gurobipy styles and
with gurobipy-pandas at several sizes, checks that their fingerprints match, and writes
the timings to `benchmarks/results/<system>.json`. `benchmarks/plot.py` draws the figures on the documentation's
[Benchmarks](docs/benchmarks.md) page from such a file and prints its table:

```shell
uv run --group bench python benchmarks/build_model.py --sizes 1000 10000 100000 1000000 10000000 --cpus 0,2,4,6,8,10
uv run --group bench python benchmarks/plot.py benchmarks/results/results.json
```

`--cpus` pins the run to one thread per performance core; on hybrid CPUs the operating system otherwise moves it
between core types. 10M rows need well over 11 GB of free memory.

`benchmarks/native.rs` counts the instructions and allocated bytes of the pure Rust stages under Valgrind, on inputs the
size of a million-term model. The `Benchmarks` workflow compares every pull request against its base commit and
writes the differences to the job summary, without failing on them. On Linux, with Valgrind 3.19 or later installed:

```shell
cargo install iai-callgrind-runner --version 0.16.1
cargo bench --bench native -- --save-baseline=base   # before a change
cargo bench --bench native -- --baseline=base        # after it
```

### Documentation

The documentation is built with [Zensical](https://zensical.org/) from `docs/`, with the API reference generated from
the docstrings:

```shell
uv run zensical serve
uv run zensical build --clean --strict
```

`serve` rebuilds on changes and serves the site on `http://localhost:8000`.

### Keep in Mind

- **gurobipy's private attributes.** polarobi registers its handles through `Model._cmodel`, `_internal_vars`,
  `_livevars`, `_constrs`, `_liveconstrs` and `_has_pending_deletions()`, so gurobipy is pinned to one minor version.
  Raising the pin needs a `thorough` run of the equivalence harness.
- **No numpy at runtime.** numpy is a dev dependency only, because polars' type annotations refer to it.
  `tests/test_no_numpy.py` fails if polarobi ever imports it.
- **The Gurobi library is never loaded twice.** The native module only ever binds to the copy gurobipy loaded.

### Layout

```
polarobi/
├── src/                      # Rust extension, built as polarobi._native
│   ├── lib.rs                # Module definition
│   ├── gurobi.rs             # Binding to gurobipy's Gurobi library, error handling
│   ├── columns.rs            # Zero-copy views of Arrow columns as C arrays
│   ├── model.rs              # Bulk adds, attribute reads and writes
│   └── handles.rs            # gurobipy handle creation
├── python/polarobi/          # Python package
│   ├── _namespace.py         # The df.grb namespace (GurobiFrame), the public API
│   ├── _api.py               # Its implementation: inputs, names, constraint layout
│   ├── _handles.py           # Registering handles with gurobipy
│   ├── _library.py           # Finding and binding the Gurobi library
│   └── _native.pyi           # Type stub of the extension
├── tests/
│   ├── harness/              # Equivalence harness: reference implementation, comparison
│   ├── test_equivalence.py   # State machine against plain gurobipy
│   └── test_*.py             # Compatibility, parity and edge cases
└── docs/                     # Documentation (Zensical)
```
