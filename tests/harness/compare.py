"""
Assert that two gurobipy models are identical.

Several independent views are compared, so that a difference shows up in a readable form and
nothing slips through any single one of them: model attributes including Gurobi's `Fingerprint`,
every variable and constraint attribute that defines the model, the matrix row by row, the MPS
file Gurobi writes, and gurobipy's Python-side bookkeeping of the handles.
"""

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gurobipy as gp

_MODEL_ATTRS = (
    "NumVars",
    "NumConstrs",
    "NumNZs",
    "NumIntVars",
    "NumBinVars",
    "ModelSense",
    "ObjCon",
    "Fingerprint",
)
_VAR_ATTRS = ("VarName", "LB", "UB", "Obj", "VType", "BranchPriority")
_CONSTR_ATTRS = ("ConstrName", "Sense", "RHS")


def assert_identical(native: gp.Model, reference: gp.Model) -> None:
    """`native` was built through polarobi, `reference` through gurobipy alone."""
    native.update()
    reference.update()
    for attr in _MODEL_ATTRS:
        got, want = native.getAttr(attr), reference.getAttr(attr)
        assert got == want, f"model attribute {attr}: {got!r} != {want!r}"
    _assert_elements(
        "variable", native, native.getVars(), reference, reference.getVars(), _VAR_ATTRS
    )
    _assert_elements(
        "constraint", native, native.getConstrs(), reference, reference.getConstrs(), _CONSTR_ATTRS
    )
    _assert_sequence("matrix row", _rows(native), _rows(reference))
    assert _mps(native) == _mps(reference), "written MPS files differ"
    _assert_bookkeeping(native, reference)


def _assert_elements(
    kind: str,
    native: gp.Model,
    native_elements: Sequence[Any],
    reference: gp.Model,
    reference_elements: Sequence[Any],
    attrs: Sequence[str],
) -> None:
    assert len(native_elements) == len(reference_elements), f"number of {kind}s"
    if not native_elements:
        return
    for attr in attrs:
        got = native.getAttr(attr, list(native_elements))
        want = reference.getAttr(attr, list(reference_elements))
        _assert_sequence(f"{kind} {attr}", got, want)


def _assert_sequence(what: str, got: Sequence[Any], want: Sequence[Any]) -> None:
    assert len(got) == len(want), f"{what}: {len(got)} != {len(want)} elements"
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        # repr() also tells 0.0 from -0.0, which compare equal.
        assert repr(g) == repr(w), f"{what} {i}: {g!r} != {w!r}"


def _rows(model: gp.Model) -> list[list[tuple[int, float]]]:
    rows: list[list[tuple[int, float]]] = []
    for constr in model.getConstrs():
        row = model.getRow(constr)
        rows.append([(row.getVar(i).index, row.getCoeff(i)) for i in range(row.size())])
    return rows


def _mps(model: gp.Model) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.mps"
        model.write(str(path))  # pyright: ignore[reportUnknownMemberType] - gurobipy's stub
        return path.read_bytes()


def _assert_bookkeeping(native: gp.Model, reference: gp.Model) -> None:
    # gurobipy's own view of the model must match what Gurobi holds, through the same private
    # state that polarobi updates when it registers handles.
    for kind, elements, pool, live in (
        ("variable", native.getVars(), native._internal_vars, native._livevars),
        ("constraint", native.getConstrs(), native._constrs, native._liveconstrs),
    ):
        assert [e.index for e in elements] == list(range(len(elements))), f"{kind} indices"
        assert len(pool) == len(elements) == live, f"gurobipy {kind} bookkeeping"
    assert native._livevars == reference._livevars
    assert native._liveconstrs == reference._liveconstrs
