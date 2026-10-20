"""
Time every call of polarobi's Python API end to end, for A/B comparisons of two builds of the
extension. Unlike `build_model.py`, which builds one realistic model, this runs each call on
inputs of `--rows` rows and prints the time of every phase as one JSON line per repeat, labelled
by `--label`:

- `add_vars`: with names, bounds and an objective, then without names
- `add_constrs`: short rows with names; wide rows whose terms repeat variables, which the
  extension has to merge; one constraint over every variable through `add_constr`
- `add_var`: one variable, the single-element path
- `set`: a float attribute from a column, a float scalar, and an int attribute from a column
- `update`: the `model.update()` that flushes the additions to Gurobi
- `get`: a float and an int attribute of every variable
- `handles`: the gurobipy handles of every variable, and a `tupledict` keyed by two columns

Timings differ by a few percent between runs on the same machine, so two builds are compared by
running them alternately, each with the other build's package shadowed through `PYTHONPATH`, and
taking the minimum of each; `.github/workflows/bench.yml` does this for every pull request. The
line also names the package that was actually imported, which guards against timing the wrong one.

    uv run --group bench python benchmarks/ab.py --rows 1000000 --label head
    PYTHONPATH=<unpacked base wheel> uv run --group bench python benchmarks/ab.py --label base
"""

import argparse
import gc
import json
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import polarobi
import polars as pl
from polarobi import GurobiFrame as grb


def inputs(rows: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`rows` keys `(i, j)` with ten `j` per `i` and random costs; one row per `i` as the groups."""
    rng = np.random.default_rng(seed)
    keys = pl.DataFrame(
        {
            "i": pl.Series(np.arange(rows) // 10, dtype=pl.Int64),
            "j": pl.Series(np.arange(rows) % 10, dtype=pl.Int64),
            "cost": pl.Series(rng.uniform(1, 10, size=rows)),
            "priority": pl.Series(rng.integers(0, 5, size=rows), dtype=pl.Int32),
        }
    )
    groups = keys.group_by("i", maintain_order=True).agg(pl.len().alias("n"))
    return keys, groups


def scenario(model: gp.Model, keys: pl.DataFrame, groups: pl.DataFrame) -> dict[str, float]:
    """Every API call in turn; the seconds each took, by phase."""
    phases: dict[str, float] = {}
    start = time.perf_counter()

    def done(phase: str) -> None:
        nonlocal start
        now = time.perf_counter()
        phases[phase] = now - start
        start = now

    x = grb(keys).add_vars(model, by=["i", "j"], lb=0.0, ub=1.0, obj="cost", vtype="B", name="x")
    y = grb(keys).add_vars(model, lb=0.0, ub=1.0, vtype="C")
    done("add_vars")
    # Short rows with names: ten terms per `i`.
    grb(x).add_constrs(model, groups, by="i", sense="<=", rhs=1.0, name="pick")
    # Wide rows with repeats: every `x` twice per `j`, so the extension merges the pairs.
    twice = pl.concat([x.select("j", "idx", "cost"), x.select("j", "idx", "cost")])
    grb(twice).add_constrs(model, by="j", sense=">=", rhs=0.0, coef="cost")
    grb(x).add_constr(model, sense=">=", rhs=1.0, name="any")
    done("add_constrs")
    polarobi.add_var(model, lb=0.0, ub=10.0, obj=1.0, name="z")
    done("add_var")
    grb(x).set(model, "Obj", "cost")
    grb(y).set(model, "UB", 2.0)
    grb(x).set(model, "BranchPriority", "priority")
    done("set")
    model.update()  # Attributes can only be read once the additions are flushed to Gurobi.
    done("update")
    grb(x).get(model, "Obj")
    grb(x).get(model, "BranchPriority", dtype=pl.Int32)
    done("get")
    grb(x).handles(model)
    grb(x).tupledict(model, ["i", "j"])
    done("handles")
    phases["total"] = sum(phases.values())
    return phases


def run(env: gp.Env, keys: pl.DataFrame, groups: pl.DataFrame) -> dict[str, float]:
    gc.collect()
    with gp.Model(env=env) as model:
        return scenario(model, keys, groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time polarobi's API calls end to end.")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="head", help="Written into every line as `build`.")
    args = parser.parse_args()

    env = gp.Env(params={"OutputFlag": 0})
    keys, groups = inputs(args.rows, args.seed)
    run(env, keys, groups)  # Warm-up, untimed: first calls pay one-off setup costs.
    package = str(Path(polarobi.__file__).parent)
    for _ in range(args.repeats):
        phases = run(env, keys, groups)
        line = {"build": args.label, "rows": args.rows, "package": package, **phases}
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
