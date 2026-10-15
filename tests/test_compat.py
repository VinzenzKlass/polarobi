# Type checkers cannot see the runtime-registered `df.grb` namespace, so its results are Unknown.
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The models built through polarobi must be indistinguishable from gurobipy-built ones."""

import gurobipy as gp
import polarobi
import polars as pl
import pytest


@pytest.fixture
def model():
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
        yield m


_keys = pl.DataFrame({"v": ["A", "A", "B", "B", "C"], "w": ["01", "02", "01", "02", "01"]})
_qty = [3.0, 1.0, 4.0, 1.0, 5.0]
_caps = pl.DataFrame({"v": ["A", "B", "C"], "cap": [3.5, 4.5, 2.0]})


def _gurobipy_reference(m: gp.Model) -> None:
    x = m.addVars(_keys.iter_rows(), ub=1.0, obj=_qty, vtype="B", name="x")
    for v, cap in _caps.iter_rows():
        terms = [(q, x[k]) for k, q in zip(_keys.iter_rows(), _qty, strict=True) if k[0] == v]
        m.addLConstr(gp.LinExpr(terms) <= cap, name=f"cap[{v}]")
    m.ModelSense = gp.GRB.MAXIMIZE


def _polarobi(m: gp.Model) -> tuple[pl.DataFrame, pl.DataFrame]:
    keys = _keys.with_columns(q=pl.Series(_qty))
    x = keys.grb.add_vars(m, by=("v", "w"), ub=1.0, obj="q", vtype="B", name="x")
    caps = x.grb.add_constrs(m, _caps, by="v", sense="<=", rhs="cap", coef="q", name="cap")
    m.ModelSense = gp.GRB.MAXIMIZE
    return x, caps


def _var(model: gp.Model, name: str) -> gp.Var:
    var = model.getVarByName(name)
    assert var is not None, name
    return var


def _constr(model: gp.Model, name: str) -> gp.Constr:
    constr = model.getConstrByName(name)
    assert constr is not None, name
    return constr


def test_identical_fingerprint_and_names() -> None:
    with (
        gp.Env(params={"OutputFlag": 0}) as env,
        gp.Model(env=env) as ref,
        gp.Model(env=env) as new,
    ):
        _gurobipy_reference(ref)
        _polarobi(new)
        ref.update()
        new.update()
        assert new.Fingerprint == ref.Fingerprint
        assert [v.VarName for v in new.getVars()] == [v.VarName for v in ref.getVars()]
        assert [c.ConstrName for c in new.getConstrs()] == [c.ConstrName for c in ref.getConstrs()]
        ref.optimize()
        new.optimize()
        assert new.ObjVal == ref.ObjVal


def test_gurobipy_api_on_native_elements(model: gp.Model) -> None:
    py = model.addVar(name="py")  # gurobipy-created, pending, before and after native ones
    x, caps = _polarobi(model)
    td = x.grb.tupledict(model, ("v", "w"))
    model.addLConstr(td["A", "01"] + py <= 1, name="mixed")  # gurobipy constraint, native var
    model.update()
    assert model.NumVars == len(model.getVars()) == 6
    assert _var(model, "x[B,02]").sameAs(td["B", "02"])
    assert _constr(model, "cap[C]").index == caps.filter(v="C").item(0, "idx")
    row = model.getRow(_constr(model, "mixed"))
    assert {row.getVar(i).VarName for i in range(row.size())} == {"x[A,01]", "py"}

    model.optimize()
    solved = x.grb.get(model, "X")
    assert solved.get_column("X").to_list() == [v.X for v in x.grb.handles(model)]
    assert caps.grb.get(model, "Slack").get_column("Slack").to_list() == pytest.approx(
        [c.Slack for c in caps.grb.handles(model, kind="constr")]
    )

    model.remove(td["A", "01"])
    model.update()
    assert td["C", "01"].index == 4 and model.NumVars == 5
    model.copy().optimize()


def test_set_attr_and_pinning(model: gp.Model) -> None:
    x, _ = _polarobi(model)
    x.filter(v="C").grb.set(model, "UB", 0.0)
    x.grb.set(model, "BranchPriority", pl.Series([1, 2, 3, 4, 5]))
    model.optimize()
    assert x.grb.get(model, "BranchPriority", dtype=pl.Int32).get_column(
        "BranchPriority"
    ).to_list() == [1, 2, 3, 4, 5]
    assert x.filter(v="C").grb.get(model, "X").get_column("X").to_list() == [0.0]


def test_terms_order_and_multichunk(model: gp.Model) -> None:
    x = _keys.grb.add_vars(model)
    terms = pl.concat([x.head(2), x.tail(3)], rechunk=False).with_columns(coef=pl.lit(2.0))
    assert terms.n_chunks() > 1
    rows = terms.grb.add_constrs(model, by="v", sense="=", rhs=1.0)
    model.update()
    assert rows.get_column("v").to_list() == ["A", "B", "C"]
    assert [model.getRow(c).size() for c in rows.grb.handles(model, kind="constr")] == [2, 2, 1]


def test_empty_rows_dropped_or_kept(model: gp.Model) -> None:
    x = _keys.filter(v="A").grb.add_vars(model).with_columns(coef=pl.lit(1.0))
    assert x.grb.add_constrs(model, _caps, by="v", sense="<", rhs="cap").height == 1
    kept = x.grb.add_constrs(model, _caps, by="v", sense="<", rhs="cap", drop_empty=False)
    model.update()
    assert kept.get_column("idx").to_list() == [1, 2, 3]
    assert [model.getRow(c).size() for c in kept.grb.handles(model, kind="constr")] == [2, 0, 0]


def test_errors(model: gp.Model) -> None:
    with pytest.raises(ValueError, match="nulls"):
        pl.DataFrame({"k": [1, 2]}).grb.add_vars(model, lb=pl.Series([0.0, None]))
    terms = pl.DataFrame({"g": [0], "idx": [7], "coef": [1.0]})  # no column 7 in the model
    with pytest.raises(gp.GurobiError):
        terms.grb.add_constrs(model, by="g", sense="<")


def test_stub_matches_runtime_namespace() -> None:
    # The `DataFrame.grb` stub claims `GurobiFrame`; the runtime-registered namespace must be it.
    assert type(pl.DataFrame({"k": [1]}).grb) is polarobi.GurobiFrame
