# Type checkers cannot see the runtime-registered `df.grb` namespace.
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
"""
Behaviour the equivalence state machine keeps out of its inputs: errors and edge cases.

Where gurobipy raises, polarobi raises the same exception type. A failing call never changes the
model: polarobi validates before the single Gurobi call and only registers handles after it.
"""

from collections.abc import Callable, Iterator

import gurobipy as gp
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.harness import native, reference
from tests.harness.compare import assert_identical
from tests.harness.specs import AddConstrs, AddVars, GetAttr
from tests.harness.strategies import key_frames

_ENV = gp.Env(params={"OutputFlag": 0, "Threads": 1})


@pytest.fixture
def models() -> Iterator[tuple[gp.Model, gp.Model]]:
    """A model for polarobi and an untouched twin it must still equal after a failed call."""
    with gp.Model(env=_ENV) as polarobi_model, gp.Model(env=_ENV) as untouched:
        for model in (polarobi_model, untouched):
            model.addVars(3, name="base")
            model.update()
        yield polarobi_model, untouched


def _both_raise(
    expected: type[BaseException],
    models: tuple[gp.Model, gp.Model],
    on_native: Callable[[gp.Model], object],
    on_gurobipy: Callable[[gp.Model], object],
) -> None:
    """Both raise `expected`; polarobi leaves its model unchanged, unlike gurobipy at times."""
    polarobi_model, untouched = models
    with gp.Model(env=_ENV) as gurobipy_model, pytest.raises(expected):
        gurobipy_model.addVars(3, name="base")
        gurobipy_model.update()
        on_gurobipy(gurobipy_model)
    with pytest.raises(expected):
        on_native(polarobi_model)
    assert_identical(polarobi_model, untouched)


def test_duplicate_keys_in_vars(models: tuple[gp.Model, gp.Model]) -> None:
    # gurobipy adds the variables before raising; polarobi adds none.
    spec = AddVars(pl.DataFrame({"k": ["a", "b", "a"]}), name="x")
    _both_raise(
        KeyError, models, lambda m: native.add_vars(m, spec), lambda m: reference.add_vars(m, spec)
    )


def test_pending_elements_have_no_attributes() -> None:
    """Like `Model.getAttr`, reading a variable that awaits `update()` raises."""
    with gp.Model(env=_ENV) as polarobi_model, gp.Model(env=_ENV) as gurobipy_model:
        pending = gurobipy_model.addVar()
        polarobi_model.addVar()
        with pytest.raises(gp.GurobiError):
            gurobipy_model.getAttr("LB", [pending])
        with pytest.raises(gp.GurobiError):
            native.get_attr(polarobi_model, GetAttr("var", "LB", pl.DataFrame({"idx": [0]})))


def test_unknown_variable_index(models: tuple[gp.Model, gp.Model]) -> None:
    terms = pl.DataFrame({"g": ["a", "b"], "idx": [0, 7], "coef": [1.0, 1.0]})
    spec = AddConstrs(terms, ("g",), "<")
    polarobi_model, gurobipy_model = models
    with pytest.raises(gp.GurobiError):
        native.add_constrs(polarobi_model, spec)
    # Nothing was added, not even the valid first row.
    assert_identical(polarobi_model, gurobipy_model)


@pytest.mark.parametrize(
    "spec",
    [
        AddVars(pl.DataFrame({"k": [1, 2]}), lb=pl.Series([0.0, None])),
        AddVars(pl.DataFrame({"k": [1, 2]}), obj=pl.Series([1.0])),
        AddVars(pl.DataFrame({"k": [1]}, schema={"k": pl.Datetime}), name="x"),
    ],
    ids=["null-bound", "short-series", "unnameable-key"],
)
def test_invalid_input_changes_nothing(models: tuple[gp.Model, gp.Model], spec: AddVars) -> None:
    polarobi_model, gurobipy_model = models
    with pytest.raises((ValueError, TypeError)):
        native.add_vars(polarobi_model, spec)
    assert_identical(polarobi_model, gurobipy_model)


@given(key_frames(max_rows=20), st.sampled_from(["x", "var", "_"]))
def test_names_match_gurobipy(keys: pl.DataFrame, name: str) -> None:
    """Every supported key dtype, nulls included, is named exactly as gurobipy names it."""
    with gp.Model(env=_ENV) as polarobi_model, gp.Model(env=_ENV) as gurobipy_model:
        spec = AddVars(keys, name=name)
        native.add_vars(polarobi_model, spec)
        reference.add_vars(gurobipy_model, spec)
        polarobi_model.update()
        assert [v.VarName for v in polarobi_model.getVars()] == [
            v.VarName for v in gurobipy_model.getVars()
        ]


def test_repeated_variables_sum_left_to_right() -> None:
    """
    Repeats are summed left to right, on every platform. gurobipy sums them in the order of an
    unstable sort, which depends on the other terms: for this row, right to left.
    """
    repeated = [-999999999.0 * 3, -183862175.19652724 * 3, 627515632.0 * 3]
    left_to_right = (repeated[0] + repeated[1]) + repeated[2]
    assert left_to_right != repeated[0] + (repeated[1] + repeated[2])
    terms = pl.DataFrame(
        {"g": ["a"] * 17, "idx": [0] * 13 + [1, 1, 1, 0], "coef": [0.0] * 13 + [*repeated, 0.0]}
    )
    spec = AddConstrs(terms, ("g",), "<=")
    with gp.Model(env=_ENV) as polarobi_model, gp.Model(env=_ENV) as gurobipy_model:
        for model in (polarobi_model, gurobipy_model):
            model.addVars(2)
        native.add_constrs(polarobi_model, spec)
        reference.add_constrs(gurobipy_model, spec)
        assert_identical(polarobi_model, gurobipy_model)
        row = polarobi_model.getRow(polarobi_model.getConstrs()[0])
        assert row.size() == 1 and row.getCoeff(0) == left_to_right


def test_failed_family_with_repeats_changes_nothing(models: tuple[gp.Model, gp.Model]) -> None:
    """A family with repeats and an invalid index is rejected as a whole."""
    terms = pl.DataFrame({"g": ["a", "a", "b"], "idx": [0, 0, 7], "coef": [1.0, 2.0, 1.0]})
    polarobi_model, untouched = models
    with pytest.raises(gp.GurobiError):
        native.add_constrs(polarobi_model, AddConstrs(terms, ("g",), "<"))
    assert_identical(polarobi_model, untouched)
