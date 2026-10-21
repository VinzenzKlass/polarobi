"""
The native constraint call on list columns that the public API never produces itself: sliced
lists, whose offsets do not start at zero, lists in several chunks, and malformed pairs.
"""

import gurobipy as gp
import polars as pl
import pytest
from polarobi import _handles, _native

_SENSE = ord("=")


@pytest.fixture
def model():
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
        m.addVars(3)
        yield m


def _add(model: gp.Model, vars_: pl.Series, coefs: pl.Series) -> list[list[tuple[int, float]]]:
    """Add one constraint per list and return each row as (variable index, coefficient) pairs."""
    n, start = vars_.len(), len(model._constrs)
    sense = pl.repeat(_SENSE, n, dtype=pl.UInt8, eager=True)
    rhs = pl.repeat(0.0, n, dtype=pl.Float64, eager=True)
    n_vars = len(model._internal_vars)  # like `_api`: pending variables count too
    _native.add_constrs(model._cmodel, vars_, coefs, sense, rhs, n_vars)
    _handles.register(model, "constr", start, n)
    model.update()
    rows = [model.getRow(c) for c in model.getConstrs()[start:]]
    return [[(r.getVar(i).index, r.getCoeff(i)) for i in range(r.size())] for r in rows]


def test_sliced_lists(model: gp.Model) -> None:
    vars_ = pl.Series([[0, 1], [2], [1, 2]], dtype=pl.List(pl.Int32)).slice(1)
    coefs = pl.Series([[1.0, 1.0], [2.0], [3.0, 4.0]], dtype=pl.List(pl.Float64)).slice(1)
    assert _add(model, vars_, coefs) == [[(2, 2.0)], [(1, 3.0), (2, 4.0)]]


def test_multichunk_lists(model: gp.Model) -> None:
    chunks = [[[0]], [[1, 2]], [[], [2, 0]]]
    vars_ = pl.concat([pl.Series(c, dtype=pl.List(pl.UInt32)) for c in chunks], rechunk=False)
    coefs = pl.concat(
        [pl.Series([[1.0]]), pl.Series([[2.0, 3.0]]), pl.Series([[], [4.0, 5.0]])], rechunk=False
    )
    assert vars_.n_chunks() == 3
    assert _add(model, vars_, coefs) == [[(0, 1.0)], [(1, 2.0), (2, 3.0)], [], [(0, 5.0), (2, 4.0)]]


def test_empty_rows_and_repeats(model: gp.Model) -> None:
    vars_ = pl.Series([[0, 0, 1], [], [2]], dtype=pl.List(pl.UInt32))
    coefs = pl.Series([[1.0, 2.0, 3.0], [], [4.0]])
    assert _add(model, vars_, coefs) == [[(0, 3.0), (1, 3.0)], [], [(2, 4.0)]]
    none = pl.Series([], dtype=pl.List(pl.UInt32)), pl.Series([], dtype=pl.List(pl.Float64))
    assert _add(model, *none) == []


def test_index_range(model: gp.Model) -> None:
    for bad in (3, -1):
        with pytest.raises(gp.GurobiError, match=f"index {bad} out of range") as info:
            _add(model, pl.Series([[0, bad]], dtype=pl.List(pl.Int32)), pl.Series([[1.0, 1.0]]))
        assert info.value.errno == 10006
    assert model.NumConstrs == 0


def test_malformed_lists(model: gp.Model) -> None:
    vars_ = pl.Series([[0], [1, 2]], dtype=pl.List(pl.UInt32))
    with pytest.raises(ValueError, match="same lengths"):
        _add(model, vars_, pl.Series([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="nulls"):
        _add(model, pl.Series([[0], None], dtype=pl.List(pl.UInt32)), pl.Series([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="nulls"):
        _add(model, vars_, pl.Series([[1.0], [2.0, None]]))
    with pytest.raises(TypeError, match="expected List"):
        _add(model, pl.Series([0, 1], dtype=pl.Int32), pl.Series([[1.0], [2.0]]))
    with pytest.raises(TypeError, match="expected Float64"):
        _add(model, vars_, pl.Series([[1], [2, 3]], dtype=pl.List(pl.Int64)))
    assert model.NumConstrs == 0


# Enough elements that the native calls create handles while Gurobi adds them.
_OVERLAPPED = 1 << 15


@pytest.mark.parametrize("n", [5, _OVERLAPPED])
def test_handles_registered_with_the_call(model: gp.Model, n: int) -> None:
    pool = model._internal_vars
    _native.add_vars(model._cmodel, n, pool=pool, cls=gp.Var)
    vars_ = pl.Series([[i % 3] for i in range(n)], dtype=pl.List(pl.UInt32))
    _native.add_constrs(
        model._cmodel,
        vars_,
        pl.Series([[1.0]] * n),
        pl.repeat(_SENSE, n, dtype=pl.UInt8, eager=True),
        pl.repeat(0.0, n, dtype=pl.Float64, eager=True),
        len(pool),
        pool=model._constrs,
        cls=gp.Constr,
    )
    assert [v._colno for v in pool] == list(range(3 + n))
    assert [c._rowno for c in model._constrs] == list(range(n))


@pytest.mark.parametrize("n", [5, _OVERLAPPED])
def test_failed_call_registers_no_handles(model: gp.Model, n: int) -> None:
    vtype = pl.repeat(ord("X"), n, dtype=pl.UInt8, eager=True)  # not a variable type
    with pytest.raises(gp.GurobiError):
        _native.add_vars(model._cmodel, n, vtype=vtype, pool=model._internal_vars, cls=gp.Var)
    assert len(model._internal_vars) == 3
