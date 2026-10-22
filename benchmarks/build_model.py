# pyright: reportUnknownMemberType=false
# gurobipy-pandas leaves its `index_formatter` callables untyped, so its functions read as
# partially unknown.
"""
Benchmark: building the same model with polarobi, plain gurobipy and gurobipy-pandas.

The model is a typical allocation model, at several sizes:

- `x`: one binary variable per (order, factory, week) option, worth its value
- `assign`: every order is chosen at most once
- `capacity`: the quantity per factory and week stays below its capacity
- `y`, `moq_min`, `moq_max`: a factory week either produces nothing or at least a minimum
  quantity, a big-M link through one binary variable per factory week

The gurobipy side is written the way such models usually are: `iter_rows()` into `addVars`, and
`group_by().agg()` with tupledict lookups into one `LinExpr` per key. Its constraints are added in
the two common ways, so there are two gurobipy builds: a loop of `addLConstr` calls, and one
`addConstrs` call per family over a generator.

A third build uses gurobipy-pandas, written as its documentation suggests: `gppd.add_vars` on
frames indexed by their keys, `groupby().sum()` into one `LinExpr` per key and `gppd.add_constrs`;
its inputs are converted to pandas before timing.

All builds produce identical models, with the same names, which is checked through Gurobi's
`Fingerprint` on every run.

Inputs are generated from a fixed seed with numpy, so runs are repeatable. Only model building is
timed, including the final `update()`; nothing is solved, so no license limits apply.

    uv run --group bench python benchmarks/build_model.py
    uv run --group bench python benchmarks/build_model.py --sizes 10000 100000 --repeats 5
    uv run --group bench python benchmarks/build_model.py --builders polarobi gurobipy_loop
"""

import argparse
import ctypes
import datetime
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import gurobipy as gp
import gurobipy_pandas as gppd
import numpy as np
import pandas as pd
import polars as pl
from polarobi import GurobiFrame as grb

type Variables = tuple[
    gp.tupledict[tuple[str, str, str], gp.Var], gp.tupledict[tuple[str, str], gp.Var]
]
type Expressions = tuple[dict[str, gp.LinExpr], dict[tuple[str, str], gp.LinExpr]]

_keys = ["factory", "week"]
_results = Path(__file__).parent / "results"
_model = {
    "variables": "rows options x plus one y per factory week",
    "constraints": "assign per order, capacity, moq_min and moq_max per factory week",
}
_pandas: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}


@dataclass(frozen=True, slots=True)
class Data:
    options: pl.DataFrame
    """One row per (order, factory, week) option with its quantity and value."""
    capacity: pl.DataFrame
    """One row per (factory, week) with its capacity and minimum order quantity."""


@dataclass(slots=True)
class Timing:
    variables: float
    constraints: float

    @property
    def total(self) -> float:
        return self.variables + self.constraints


def generate(rows: int, seed: int) -> Data:
    """`rows` options, about four per order, over 50 factories and 52 weeks."""
    # Vectorized, so that generating ten million rows takes seconds and little memory.
    rng = np.random.default_rng(seed)
    factories, weeks = 50, 52
    orders = max(1, rows // 4)
    combinations = orders * factories * weeks
    # Draw a few spare keys, as some repeat, and keep the first `rows` distinct ones in order.
    keys = pl.Series("key", rng.integers(0, combinations, size=rows + rows // 50 + 100))
    keys = keys.unique(maintain_order=True).head(rows)
    if keys.len() < rows:
        raise ValueError(f"{rows:,} options do not fit {combinations:,} combinations")
    quantity = rng.integers(1, 501, size=rows)
    frame = (
        keys.to_frame()
        .with_columns(
            quantity=pl.Series(quantity, dtype=pl.Int64),
            value=pl.Series(rng.uniform(1, 10, size=rows) * quantity),
        )
        .select(
            order=pl.concat_str(
                pl.lit("O"), (pl.col("key") // (factories * weeks)).cast(pl.String)
            ),
            factory=pl.concat_str(
                pl.lit("F"), (pl.col("key") // weeks % factories).cast(pl.String)
            ),
            week=pl.concat_str(pl.lit("W"), (pl.col("key") % weeks).cast(pl.String)),
            quantity="quantity",
            value="value",
        )
    )
    capacity = (
        frame.group_by(_keys, maintain_order=True)
        .agg(pl.sum("quantity"))
        .select(
            *_keys,
            capacity=(pl.col("quantity") * 0.4).round(),
            moq=(pl.col("quantity") * 0.1).round(),
        )
    )
    return Data(frame, capacity)


def _gurobipy_variables(model: gp.Model, data: Data) -> Variables:
    options = data.options
    x = model.addVars(
        options.select("order", *_keys).iter_rows(),
        vtype=gp.GRB.BINARY,
        obj=options.get_column("value").to_list(),
        name="x",
    )
    y = model.addVars(data.capacity.select(_keys).iter_rows(), vtype=gp.GRB.BINARY, name="y")
    return x, y


def _gurobipy_expressions(data: Data, x: gp.tupledict[tuple[str, str, str], gp.Var]) -> Expressions:
    """One expression per key: the orders' assignments, and the quantity per factory week."""
    assign: dict[str, gp.LinExpr] = {}
    grouped = data.options.group_by("order", maintain_order=True).agg(*_keys)
    for order, factories, weeks in grouped.iter_rows():
        terms = [x[order, f, w] for f, w in zip(factories, weeks, strict=True)]
        assign[order] = gp.LinExpr([1.0] * len(terms), terms)
    quantity: dict[tuple[str, str], gp.LinExpr] = {}
    grouped = data.options.group_by(_keys, maintain_order=True).agg("order", "quantity")
    for factory, week, orders, quantities in grouped.iter_rows():
        terms = [x[o, factory, week] for o in orders]
        quantity[factory, week] = gp.LinExpr(quantities, terms)
    return assign, quantity


def build_gurobipy_loop(model: gp.Model, data: Data) -> Timing:
    """Constraints added one by one with `addLConstr`."""
    start = time.perf_counter()
    x, y = _gurobipy_variables(model, data)
    variables = time.perf_counter()

    assign, quantity = _gurobipy_expressions(data, x)
    for order, expr in assign.items():
        model.addLConstr(expr, "<", 1.0, name=f"assign[{order}]")
    for factory, week, cap, _ in data.capacity.iter_rows():
        model.addLConstr(quantity[factory, week], "<", cap, name=f"capacity[{factory},{week}]")
    for factory, week, _, moq in data.capacity.iter_rows():
        expr = quantity[factory, week] - moq * y[factory, week]
        model.addLConstr(expr, ">", 0.0, name=f"moq_min[{factory},{week}]")
    for factory, week, cap, _ in data.capacity.iter_rows():
        expr = quantity[factory, week] - cap * y[factory, week]
        model.addLConstr(expr, "<", 0.0, name=f"moq_max[{factory},{week}]")
    model.update()
    return Timing(variables - start, time.perf_counter() - variables)


def build_gurobipy_addconstrs(model: gp.Model, data: Data) -> Timing:
    """Constraints added per family with `addConstrs` over a generator."""
    start = time.perf_counter()
    x, y = _gurobipy_variables(model, data)
    variables = time.perf_counter()

    # addConstrs names each constraint by its generator's loop variables, so the generators loop
    # over the keys only.
    assign, quantity = _gurobipy_expressions(data, x)
    model.addConstrs((assign[order] <= 1.0 for order in assign), name="assign")
    limits = {(f, w): (cap, moq) for f, w, cap, moq in data.capacity.iter_rows()}
    model.addConstrs((quantity[f, w] <= limits[f, w][0] for f, w in limits), name="capacity")
    model.addConstrs(
        (quantity[f, w] - limits[f, w][1] * y[f, w] >= 0.0 for f, w in limits), name="moq_min"
    )
    model.addConstrs(
        (quantity[f, w] - limits[f, w][0] * y[f, w] <= 0.0 for f, w in limits), name="moq_max"
    )
    model.update()
    return Timing(variables - start, time.perf_counter() - variables)


def build_polarobi(model: gp.Model, data: Data) -> Timing:
    start = time.perf_counter()
    x = grb(data.options).add_vars(model, by=["order", *_keys], vtype="B", obj="value", name="x")
    y = grb(data.capacity).add_vars(model, by=_keys, vtype="B", name="y")
    variables = time.perf_counter()

    grb(x).add_constrs(model, by="order", sense="<=", rhs=1.0, name="assign")
    quantity = x.select(*_keys, "idx", "quantity")
    grb(quantity).add_constrs(
        model, data.capacity, by=_keys, sense="<=", rhs="capacity", coef="quantity", name="capacity"
    )
    links: tuple[tuple[str, Literal[">=", "<="], str], ...] = (
        ("moq", ">=", "moq_min"),
        ("capacity", "<=", "moq_max"),
    )
    for linked, sense, name in links:
        link = y.select(*_keys, "idx", quantity=-pl.col(linked))
        terms = pl.concat([quantity, link], how="vertical_relaxed")
        grb(terms).add_constrs(
            model, data.capacity, by=_keys, sense=sense, rhs=0.0, coef="quantity", name=name
        )
    model.update()
    return Timing(variables - start, time.perf_counter() - variables)


def _as_pandas(data: Data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The inputs as pandas frames indexed by their keys, converted once per data set."""
    if id(data) not in _pandas:
        _pandas.clear()  # One size at a time: at 10M rows the copy takes gigabytes.
        options = data.options.to_pandas().set_index(["order", *_keys])
        _pandas[id(data)] = (options, data.capacity.to_pandas().set_index(_keys))
    return _pandas[id(data)]


def build_gurobipy_pandas(model: gp.Model, data: Data) -> Timing:
    """gurobipy-pandas as documented; its pandas inputs are prepared before timing starts."""
    options, capacity = _as_pandas(data)
    start = time.perf_counter()
    x = gppd.add_vars(model, options, vtype=gp.GRB.BINARY, obj="value", name="x")
    y = gppd.add_vars(model, capacity, vtype=gp.GRB.BINARY, name="y")
    variables = time.perf_counter()

    # sort=False keeps the keys in order of first appearance, as in the other builds.
    assign = x.groupby(level="order", sort=False).sum()
    gppd.add_constrs(model, assign, gp.GRB.LESS_EQUAL, 1.0, name="assign")
    terms = x * options["quantity"]
    quantity = terms.groupby(level=_keys, sort=False).sum().reindex(capacity.index)
    gppd.add_constrs(model, quantity, gp.GRB.LESS_EQUAL, capacity["capacity"], name="capacity")
    low = quantity - capacity["moq"] * y
    gppd.add_constrs(model, low, gp.GRB.GREATER_EQUAL, 0.0, name="moq_min")
    high = quantity - capacity["capacity"] * y
    gppd.add_constrs(model, high, gp.GRB.LESS_EQUAL, 0.0, name="moq_max")
    model.update()
    return Timing(variables - start, time.perf_counter() - variables)


def run(
    env: gp.Env, data: Data, builder: Callable[[gp.Model, Data], Timing], *, names: bool = False
) -> tuple[Timing, int, str]:
    """
    Time `builder` on a fresh model; return its timing, the model's fingerprint and, with `names`,
    a digest of all variable and constraint names, which the fingerprint does not cover.
    """
    gc.collect()
    with gp.Model(env=env) as model:
        timing = builder(model, data)
        fingerprint = model.Fingerprint
        digest = ""
        if names:
            text = "\n".join([*model.getAttr("VarName"), *model.getAttr("ConstrName")])
            digest = hashlib.sha256(text.encode()).hexdigest()
    return timing, fingerprint, digest


def _pin(cpus: list[int]) -> None:
    """Pin this process to the logical CPUs `cpus`, with the operating system's own call."""
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        mask = sum(1 << cpu for cpu in cpus)
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    else:
        raise SystemExit("--cpus is not supported on this platform")


def main() -> None:
    builders: dict[str, Callable[[gp.Model, Data], Timing]] = {
        "gurobipy_loop": build_gurobipy_loop,
        "gurobipy_addconstrs": build_gurobipy_addconstrs,
        "gurobipy_pandas": build_gurobipy_pandas,
        "polarobi": build_polarobi,
    }
    parser = argparse.ArgumentParser(description="Benchmark model building against polarobi.")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000, 10_000_000]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=_results / f"{platform.system().lower()}.json")
    parser.add_argument(
        "--cpus",
        type=lambda text: [int(cpu) for cpu in text.split(",")],
        help="Logical CPUs to pin the process to, e.g. 0,2,4,6,8,10. On hybrid CPUs, pick one "
        "thread per performance core: otherwise the OS moves the run between core types.",
    )
    parser.add_argument(
        "--builders",
        nargs="+",
        choices=builders,
        default=list(builders),
        help="Builds to run, all by default. Models are only compared between the builds run.",
    )
    args = parser.parse_args()
    if args.cpus:
        _pin(args.cpus)
    builders = {name: builders[name] for name in dict.fromkeys(args.builders)}

    env = gp.Env(params={"OutputFlag": 0})
    results: list[dict[str, object]] = []
    for size in args.sizes:
        data = generate(size, args.seed)
        fingerprints: set[int] = set()
        digests: dict[str, str] = {}
        for name, builder in builders.items():
            # Warm-up, untimed: first calls pay one-off setup costs. It also checks the names.
            _, _, digests[name] = run(env, data, builder, names=True)
            timings: list[Timing] = []
            for _ in range(args.repeats):
                timing, fingerprint, _ = run(env, data, builder)
                timings.append(timing)
                fingerprints.add(fingerprint)
            # The median run by total time, with its phases.
            median = sorted(timings, key=lambda t: t.total)[len(timings) // 2]
            results.append(
                {
                    "rows": size,
                    "builder": name,
                    "variables_s": median.variables,
                    "constraints_s": median.constraints,
                    "total_s": median.total,
                    "runs_total_s": [t.total for t in timings],
                }
            )
            print(
                f"{size:>9,} rows  {name:19}  total {median.total:7.3f} s  "
                f"(variables {median.variables:6.3f} s, constraints {median.constraints:6.3f} s, "
                f"spread {statistics.pstdev([t.total for t in timings]):.3f} s)"
            )
        if len(fingerprints) != 1:
            raise SystemExit(f"Models differ at {size:,} rows: fingerprints {sorted(fingerprints)}")
        if len(set(digests.values())) != 1:
            raise SystemExit(f"Names differ at {size:,} rows: {digests}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "date": datetime.date.today().isoformat(),
        "machine": {
            "system": f"{platform.system()} {platform.release()}",
            "processor": platform.processor() or platform.machine(),
            "cpus": os.cpu_count(),
        },
        "versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ("polarobi", "polars", "gurobipy", "gurobipy-pandas", "pandas")
            },
        },
        "settings": {
            "seed": args.seed,
            "repeats": args.repeats,
            "cpus": args.cpus,
            "builders": list(builders),
        },
        "model": _model,
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Identical fingerprints and names at every size. Results written to {args.out}.")


if __name__ == "__main__":
    main()
