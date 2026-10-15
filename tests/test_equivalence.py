"""
polarobi must build exactly the model gurobipy builds.

A Hypothesis state machine drives two models through the same random sequence of operations:
`native` through polarobi's `df.grb`, `reference` through plain gurobipy (`harness.reference`).
Plain gurobipy calls are interleaved on both, so polarobi is exercised next to pending gurobipy
variables, constraints and deletions. Every return value must match, and the models must be
identical (`harness.compare`) whenever Hypothesis chooses to check and at the end of every run.

Run more examples with `HYPOTHESIS_PROFILE=ci` or `thorough`, see `tests/conftest.py`.
"""

import gurobipy as gp
import polars as pl
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    precondition,
    rule,
    run_state_machine_as_test,  # pyright: ignore[reportUnknownVariableType] - hypothesis' stub
)
from polars.testing import assert_frame_equal, assert_series_equal

from tests.harness import native, reference, strategies
from tests.harness.compare import assert_identical
from tests.harness.specs import Kind

_ENV = gp.Env(params={"OutputFlag": 0, "Threads": 1, "TimeLimit": 30})
# The size-limited license caps models at 2000 variables and constraints.
_MAX_ELEMENTS = 1500


class Equivalence(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.native = gp.Model(env=_ENV)
        self.reference = gp.Model(env=_ENV)

    def count(self, kind: Kind) -> int:
        # The reference model is always up to date.
        return self.reference.NumVars if kind == "var" else self.reference.NumConstrs

    def settle(self) -> None:
        """Apply pending changes, so Python-side handle positions are column/row indices."""
        self.native.update()

    # polarobi

    @precondition(lambda self: self.count("var") < _MAX_ELEMENTS)
    @rule(data=st.data())
    def add_vars(self, data: st.DataObject) -> None:
        spec = data.draw(strategies.add_vars_specs())
        want = reference.add_vars(self.reference, spec)
        got = native.add_vars(self.native, spec)
        assert_frame_equal(got, want)

    @precondition(lambda self: 0 < self.count("var") and self.count("constr") < _MAX_ELEMENTS)
    @rule(data=st.data())
    def add_constrs(self, data: st.DataObject) -> None:
        spec = data.draw(strategies.add_constrs_specs(self.count("var")))
        want = reference.add_constrs(self.reference, spec)
        got = native.add_constrs(self.native, spec)
        assert_frame_equal(got, want)

    @precondition(lambda self: self.count("var") > 0)
    @rule(data=st.data(), kind=st.sampled_from(["var", "constr"]))
    def set_attr(self, data: st.DataObject, kind: Kind) -> None:
        if self.count(kind) == 0:
            return
        spec = data.draw(strategies.set_attr_specs(kind, self.count(kind)))
        reference.set_attr(self.reference, spec)
        native.set_attr(self.native, spec)

    @precondition(lambda self: self.count("var") > 0)
    @rule(data=st.data(), kind=st.sampled_from(["var", "constr"]))
    def get_attr(self, data: st.DataObject, kind: Kind) -> None:
        if self.count(kind) == 0:
            return
        spec = data.draw(strategies.get_attr_specs(kind, self.count(kind)))
        # Pending elements have no attributes yet, gurobipy raises for them too (see
        # test_parity.py); the reference is always up to date.
        self.settle()
        assert_series_equal(
            native.get_attr(self.native, spec), reference.get_attr(self.reference, spec)
        )

    @precondition(lambda self: self.count("var") > 0)
    @rule(data=st.data(), kind=st.sampled_from(["var", "constr"]))
    def handles(self, data: st.DataObject, kind: Kind) -> None:
        if self.count(kind) == 0:
            return
        index = data.draw(strategies.index_frames(self.count(kind)))
        got = native.handles(self.native, index, kind)
        self.settle()
        want = reference.handles(self.reference, index, kind)
        assert [h.index for h in got] == [h.index for h in want]
        pool = self.native.getVars() if kind == "var" else self.native.getConstrs()
        assert all(h.sameAs(pool[h.index]) for h in got)

    # Plain gurobipy on both models, interleaved with polarobi.

    @precondition(lambda self: self.count("var") < _MAX_ELEMENTS)
    @rule(
        lb=st.sampled_from([0.0, -1.0]),
        ub=st.sampled_from([1.0, gp.GRB.INFINITY]),
        obj=st.floats(-10, 10),
    )
    def gurobipy_var(self, lb: float, ub: float, obj: float) -> None:
        for model in (self.native, self.reference):
            model.addVar(lb=lb, ub=ub, obj=obj, name="py")
        self.reference.update()

    @precondition(lambda self: 0 < self.count("var") and self.count("constr") < _MAX_ELEMENTS)
    @rule(data=st.data())
    def gurobipy_constr(self, data: st.DataObject) -> None:
        index = data.draw(strategies.index_frames(self.count("var")))
        rhs = data.draw(st.floats(-10, 10))
        # Native handles come from polarobi, so pending polarobi variables are used as they are.
        for model, side in ((self.native, native), (self.reference, reference)):
            terms: list[gp.Var] = side.handles(model, index, "var")
            model.addLConstr(gp.LinExpr([1.0] * len(terms), terms), "<", rhs, name="pyc")
        self.reference.update()

    @precondition(lambda self: self.count("var") > 0)
    @rule(data=st.data(), kind=st.sampled_from(["var", "constr"]))
    def remove(self, data: st.DataObject, kind: Kind) -> None:
        """Removals stay pending in the native model until something updates it."""
        if self.count(kind) == 0:
            return
        index = data.draw(strategies.index_frames(self.count(kind))).unique()
        self.settle()
        for model, side in ((self.native, native), (self.reference, reference)):
            model.remove(side.handles(model, index, kind))
        self.reference.update()

    @rule()
    def update(self) -> None:
        self.settle()

    @precondition(lambda self: self.count("var") > 0)
    @rule(sense=st.sampled_from([gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE]))
    def optimize(self, sense: int) -> None:
        # Solve from scratch: Gurobi warm-starts from earlier solves, and the two models reached
        # the same state through differently batched updates. The model itself is compared by
        # `check`; this confirms that identical models solve identically.
        for model in (self.native, self.reference):
            model.ModelSense = sense
            model.reset()
            model.optimize()
        assert self.native.Status == self.reference.Status
        assert self.native.SolCount == self.reference.SolCount
        if self.reference.SolCount:
            assert self.native.ObjVal == self.reference.ObjVal
            index = pl.DataFrame({"idx": range(self.count("var"))})
            spec = strategies.GetAttr("var", "X", index)
            assert_series_equal(
                native.get_attr(self.native, spec), reference.get_attr(self.reference, spec)
            )

    @rule()
    def check(self) -> None:
        assert_identical(self.native, self.reference)

    def teardown(self) -> None:
        try:
            assert_identical(self.native, self.reference)
        finally:
            self.native.dispose()
            self.reference.dispose()


def test_equivalence() -> None:
    run_state_machine_as_test(Equivalence)
