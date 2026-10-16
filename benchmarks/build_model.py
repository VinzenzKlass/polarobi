"""
Benchmark: building the same model with plain gurobipy and with polarobi.

The model is a typical allocation model, at several sizes:

- `x`: one binary variable per (order, factory, week) option, worth its value
- `assign`: every order is chosen at most once
- `capacity`: the quantity per factory and week stays below its capacity
- `y`, `moq_min`, `moq_max`: a factory week either produces nothing or at least a minimum
  quantity, a big-M link through one binary variable per factory week

The gurobipy side is written the way such models usually are: `iter_rows()` into `addVars`, and
`group_by().agg()` with tupledict lookups into one `LinExpr` per key. Its constraints are added in
the two common ways, so there are two gurobipy builds: a loop of `addLConstr` calls, and one
`addConstrs` call per family over a generator. All builds produce identical models, which is
checked through Gurobi's `Fingerprint` on every run.

Inputs are generated from a fixed seed with numpy, so runs are repeatable. Only model building is
timed, including the final `update()`; nothing is solved, so no license limits apply.

    uv run python benchmarks/build_model.py
    uv run python benchmarks/build_model.py --sizes 10000 100000 --repeats 5
"""

import argparse
import ctypes
import datetime
import gc
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
import numpy as np
import polars as pl
from polarobi import GurobiFrame as grb

_KEYS = ["factory", "week"]
_RESULTS = Path(__file__).parent / "results"


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
        frame.group_by(_KEYS, maintain_order=True)
        .agg(pl.sum("quantity"))
        .select(
            *_KEYS,
            capacity=(pl.col("quantity") * 0.4).round(),
            moq=(pl.col("quantity") * 0.1).round(),
        )
    )
    return Data(frame, capacity)


type Variables = tuple[
    gp.tupledict[tuple[str, str, str], gp.Var], gp.tupledict[tuple[str, str], gp.Var]
]
type Expressions = tuple[dict[str, gp.LinExpr], dict[tuple[str, str], gp.LinExpr]]


def _gurobipy_variables(model: gp.Model, data: Data) -> Variables:
    options = data.options
    x = model.addVars(
        options.select("order", *_KEYS).iter_rows(),
        vtype=gp.GRB.BINARY,
        obj=options.get_column("value").to_list(),
        name="x",
    )
    y = model.addVars(data.capacity.select(_KEYS).iter_rows(), vtype=gp.GRB.BINARY, name="y")
    return x, y


def _gurobipy_expressions(data: Data, x: gp.tupledict[tuple[str, str, str], gp.Var]) -> Expressions:
    """One expression per key: the orders' assignments, and the quantity per factory week."""
    assign: dict[str, gp.LinExpr] = {}
    grouped = data.options.group_by("order", maintain_order=True).agg(*_KEYS)
    for order, factories, weeks in grouped.iter_rows():
        terms = [x[order, f, w] for f, w in zip(factories, weeks, strict=True)]
        assign[order] = gp.LinExpr([1.0] * len(terms), terms)
    quantity: dict[tuple[str, str], gp.LinExpr] = {}
    grouped = data.options.group_by(_KEYS, maintain_order=True).agg("order", "quantity")
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
    x = grb(data.options).add_vars(model, by=["order", *_KEYS], vtype="B", obj="value", name="x")
    y = grb(data.capacity).add_vars(model, by=_KEYS, vtype="B", name="y")
    variables = time.perf_counter()

    grb(x).add_constrs(model, by="order", sense="<=", rhs=1.0, name="assign")
    quantity = x.select(*_KEYS, "idx", "quantity")
    grb(quantity).add_constrs(
        model, data.capacity, by=_KEYS, sense="<=", rhs="capacity", coef="quantity", name="capacity"
    )
    links: tuple[tuple[str, Literal[">=", "<="], str], ...] = (
        ("moq", ">=", "moq_min"),
        ("capacity", "<=", "moq_max"),
    )
    for linked, sense, name in links:
        link = y.select(*_KEYS, "idx", quantity=-pl.col(linked))
        terms = pl.concat([quantity, link], how="vertical_relaxed")
        grb(terms).add_constrs(
            model, data.capacity, by=_KEYS, sense=sense, rhs=0.0, coef="quantity", name=name
        )
    model.update()
    return Timing(variables - start, time.perf_counter() - variables)


_BUILDERS: dict[str, Callable[[gp.Model, Data], Timing]] = {
    "gurobipy_loop": build_gurobipy_loop,
    "gurobipy_addconstrs": build_gurobipy_addconstrs,
    "polarobi": build_polarobi,
}


def run(env: gp.Env, data: Data, builder: Callable[[gp.Model, Data], Timing]) -> tuple[Timing, int]:
    gc.collect()
    with gp.Model(env=env) as model:
        timing = builder(model, data)
        fingerprint = model.Fingerprint
    return timing, fingerprint


_MODEL = {
    "variables": "rows options x plus one y per factory week",
    "constraints": "assign per order, capacity, moq_min and moq_max per factory week",
}


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
    parser = argparse.ArgumentParser(description="Benchmark model building, gurobipy vs polarobi.")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000, 10_000_000]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=_RESULTS / f"{platform.system().lower()}.json")
    parser.add_argument(
        "--cpus",
        type=lambda text: [int(cpu) for cpu in text.split(",")],
        help="Logical CPUs to pin the process to, e.g. 0,2,4,6,8,10. On hybrid CPUs, pick one "
        "thread per performance core: otherwise the OS moves the run between core types.",
    )
    args = parser.parse_args()
    if args.cpus:
        _pin(args.cpus)

    env = gp.Env(params={"OutputFlag": 0})
    results: list[dict[str, object]] = []
    for size in args.sizes:
        data = generate(size, args.seed)
        fingerprints: set[int] = set()
        for name, builder in _BUILDERS.items():
            run(env, data, builder)  # Warm-up, untimed: first calls pay one-off setup costs.
            timings: list[Timing] = []
            for _ in range(args.repeats):
                timing, fingerprint = run(env, data, builder)
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
            **{package: version(package) for package in ("polarobi", "polars", "gurobipy")},
        },
        "settings": {"seed": args.seed, "repeats": args.repeats, "cpus": args.cpus},
        "model": _MODEL,
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Identical fingerprints at every size. Results written to {args.out}.")


if __name__ == "__main__":
    main()
