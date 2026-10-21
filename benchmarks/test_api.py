# pytest-codspeed types `pedantic`'s `setup` as a bare `Callable`, so the method reads as
# partially unknown.
# pyright: reportUnknownMemberType=false
"""
CodSpeed benchmarks of polarobi's Python API, one per call, on inputs of `ROWS` rows. Each builds
its model and inputs in `setup`, which is not measured, and measures the one call. The `CodSpeed`
workflow runs them on every pull request and on master, and CodSpeed reports the changes.

    uv run --group dev --group codspeed pytest benchmarks/ --codspeed

Outside `pytest benchmarks/`, these are not collected: the test suite's paths are `tests/` only.
"""

from collections.abc import Callable, Iterator
from typing import Any

import gurobipy as gp
import polarobi
import polars as pl
import pytest
from polarobi import GurobiFrame as grb
from pytest_codspeed import BenchmarkFixture

ROWS = 100_000

type Setup = Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]


@pytest.fixture(scope="module")
def env() -> Iterator[gp.Env]:
    with gp.Env(params={"OutputFlag": 0}) as env:
        yield env


@pytest.fixture(scope="module")
def keys() -> pl.DataFrame:
    """`ROWS` keys `(i, j)` with ten `j` per `i`, a cost and an integer priority per key."""
    return pl.DataFrame(
        {"i": pl.int_range(ROWS, eager=True) // 10, "j": pl.int_range(ROWS, eager=True) % 10}
    ).with_columns(
        cost=1.0 + (pl.col("i") * 7 + pl.col("j") * 3) % 10,
        priority=(pl.col("i") % 5).cast(pl.Int32),
    )


def fresh(env: gp.Env) -> Setup:
    """A setup that creates an empty model."""
    return lambda: ((gp.Model(env=env),), {})


def with_vars(env: gp.Env, keys: pl.DataFrame, update: bool = False) -> Setup:
    """A setup that adds the named variables of `keys` to a fresh model, and its `update()`."""

    def setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        model = gp.Model(env=env)
        x = grb(keys).add_vars(model, by=["i", "j"], ub=1.0, obj="cost", vtype="B", name="x")
        if update:
            model.update()
        return (model, x), {}

    return setup


def test_add_vars_named(benchmark: BenchmarkFixture, env: gp.Env, keys: pl.DataFrame) -> None:
    def add(model: gp.Model) -> None:
        grb(keys).add_vars(model, by=["i", "j"], ub=1.0, obj="cost", vtype="B", name="x")

    benchmark.pedantic(add, setup=fresh(env))


def test_add_vars_unnamed(benchmark: BenchmarkFixture, env: gp.Env, keys: pl.DataFrame) -> None:
    def add(model: gp.Model) -> None:
        grb(keys).add_vars(model, ub=1.0, vtype="C")

    benchmark.pedantic(add, setup=fresh(env))


def test_add_var(benchmark: BenchmarkFixture, env: gp.Env) -> None:
    def add(model: gp.Model) -> None:
        polarobi.add_var(model, ub=10.0, obj=1.0, name="z")

    benchmark.pedantic(add, setup=fresh(env))


def test_add_constrs_short_rows(
    benchmark: BenchmarkFixture, env: gp.Env, keys: pl.DataFrame
) -> None:
    """One named constraint per `i`, of ten terms each."""

    def add(model: gp.Model, x: pl.DataFrame) -> None:
        grb(x).add_constrs(model, by="i", sense="<=", rhs=1.0, name="pick")

    benchmark.pedantic(add, setup=with_vars(env, keys))


def test_add_constrs_repeats(benchmark: BenchmarkFixture, env: gp.Env, keys: pl.DataFrame) -> None:
    """Ten wide constraints, one per `j`, in which every variable appears twice, so the
    extension merges the repeats."""

    def add(model: gp.Model, x: pl.DataFrame) -> None:
        twice = pl.concat([x.select("j", "idx", "cost"), x.select("j", "idx", "cost")])
        grb(twice).add_constrs(model, by="j", sense=">=", coef="cost")

    benchmark.pedantic(add, setup=with_vars(env, keys))


def test_add_constr(benchmark: BenchmarkFixture, env: gp.Env, keys: pl.DataFrame) -> None:
    """One constraint over every variable."""

    def add(model: gp.Model, x: pl.DataFrame) -> None:
        grb(x).add_constr(model, sense=">=", rhs=1.0, name="any")

    benchmark.pedantic(add, setup=with_vars(env, keys))


def test_set(benchmark: BenchmarkFixture, env: gp.Env, keys: pl.DataFrame) -> None:
    """A float attribute from a column, one from a scalar, and an int attribute from a column."""

    def set_attrs(model: gp.Model, x: pl.DataFrame) -> None:
        grb(x).set(model, "Obj", "cost")
        grb(x).set(model, "UB", 2.0)
        grb(x).set(model, "BranchPriority", "priority")

    benchmark.pedantic(set_attrs, setup=with_vars(env, keys))


def test_get(benchmark: BenchmarkFixture, env: gp.Env, keys: pl.DataFrame) -> None:
    """A float and an int attribute of every variable."""

    def get_attrs(model: gp.Model, x: pl.DataFrame) -> None:
        grb(x).get(model, "Obj")
        grb(x).get(model, "BranchPriority", dtype=pl.Int32)

    benchmark.pedantic(get_attrs, setup=with_vars(env, keys, update=True))


def test_handles(benchmark: BenchmarkFixture, env: gp.Env, keys: pl.DataFrame) -> None:
    def handles(model: gp.Model, x: pl.DataFrame) -> None:
        grb(x).handles(model)

    benchmark.pedantic(handles, setup=with_vars(env, keys, update=True))


def test_tupledict(benchmark: BenchmarkFixture, env: gp.Env, keys: pl.DataFrame) -> None:
    def tupledict(model: gp.Model, x: pl.DataFrame) -> None:
        grb(x).tupledict(model, ["i", "j"])

    benchmark.pedantic(tupledict, setup=with_vars(env, keys, update=True))
