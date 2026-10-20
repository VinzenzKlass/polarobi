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
    rows = terms.grb.add_constrs(model, by="v", sense="=", rhs=1.0, coef="coef")
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


def test_null_keys_group_like_gurobipy(model: gp.Model) -> None:
    """A null key is a key: gurobipy groups `None` keys too, and names them `None`."""
    x = pl.DataFrame({"k": [1, 2, 3]}).grb.add_vars(model, name="x")
    terms = x.with_columns(g=pl.Series([1, None, None]), coef=pl.Series([1.0, 2.0, 3.0]))
    rows = terms.grb.add_constrs(model, by="g", sense="<=", rhs=1.0, coef="coef", name="c")
    model.update()
    assert rows.get_column("g").to_list() == [1, None]
    assert [c.ConstrName for c in rows.grb.handles(model, kind="constr")] == ["c[1]", "c[None]"]
    assert model.getRow(_constr(model, "c[None]")).size() == 2
    # Given rows match a null key as well, with or without terms.
    given = pl.DataFrame({"g": [None, 1, None, 7]})
    kept = terms.grb.add_constrs(model, given, by="g", sense="<=", rhs=1.0, drop_empty=False)
    model.update()
    assert kept.get_column("g").to_list() == [None, 1, None, 7]
    assert [model.getRow(c).size() for c in kept.grb.handles(model, kind="constr")] == [2, 1, 2, 0]
    dropped = terms.grb.add_constrs(model, given, by="g", sense="<=", rhs=1.0)
    assert dropped.get_column("g").to_list() == [None, 1, None]


def test_stub_matches_runtime_namespace() -> None:
    # `grb(df)` relies on the runtime-registered namespace being `GurobiFrame`.
    assert type(pl.DataFrame({"k": [1]}).grb) is polarobi.GurobiFrame


def test_add_var(model: gp.Model) -> None:
    model.addVar(name="before")
    unnamed = polarobi.add_var(model)
    air = polarobi.add_var(
        model, lb=0.0, ub=250.0, obj=-10.0, vtype="C", name="air_expansion", idx="col"
    )
    assert unnamed.schema == {"idx": pl.UInt32}
    assert unnamed.item(0, "idx") == 1
    assert air.item(0, "col") == 2
    assert model.NumVars == 0  # no update() performed

    constr = polarobi.add_constr(
        model, [unnamed, (air, -1.0, "col")], sense=">=", rhs=0.0, name="link"
    )
    air.grb.set(model, "UB", 300.0, idx="col")

    model.ModelSense = gp.GRB.MINIMIZE
    model.optimize()
    v_unnamed, v_air = unnamed.grb.handles(model)[0], air.grb.handles(model, idx="col")[0]
    assert v_unnamed.index == unnamed.item(0, "idx") == 1
    assert v_air.index == air.item(0, "col") == 2
    assert v_unnamed.VarName == "C1"
    assert v_air.VarName == "air_expansion"
    assert unnamed.grb.get(model, "X").item(0, "X") == pytest.approx(300.0)
    assert air.grb.get(model, "X", idx="col").item(0, "X") == pytest.approx(300.0)
    assert constr.grb.get(model, "Slack").item(0, "Slack") == pytest.approx(0.0)


def test_add_constr(model: gp.Model) -> None:
    keys = _keys.with_columns(q=pl.Series(_qty))
    x = polarobi.GurobiFrame(keys).add_vars(model, by=("v", "w"), ub=1.0, obj="q", name="x")
    cap = polarobi.add_var(model, ub=10.0, name="cap", idx="col")

    single = x.grb.add_constr(model, sense="<=", rhs=12.0, coef="q", name="total", idx="c_idx")
    multi = polarobi.add_constr(
        model, [(x, "q"), (cap, -1.0, "col")], sense="<=", rhs=0.0, name="link"
    )
    model.update()

    assert single.schema == {"c_idx": pl.UInt32}
    assert single.item(0, "c_idx") == 0
    assert multi.schema == {"idx": pl.UInt32}
    assert multi.item(0, "idx") == 1
    c_single = _constr(model, "total")
    c_multi = _constr(model, "link")
    assert c_single.index == 0 and c_single.RHS == 12.0
    assert c_multi.index == 1 and c_multi.RHS == 0.0
    assert model.getRow(c_single).size() == 5
    assert model.getRow(c_multi).size() == 6


def test_multi_source_add_constrs() -> None:
    keys = ("v", "w")
    x_dist = pl.DataFrame(
        {"v": ["A", "B"], "w": [1, 2], "quantity": [3.0, 4.0], "extra": ["x", "y"]},
        schema={"v": pl.String, "w": pl.Int32, "quantity": pl.Float64, "extra": pl.String},
    )
    x_nd = pl.DataFrame(
        {"v": ["A", "C"], "w": [1, 1], "quantity": [2, 5], "other": [10, 20]},
        schema={"v": pl.String, "w": pl.Int64, "quantity": pl.Int64, "other": pl.Int64},
    )
    breach = pl.DataFrame(
        {"v": ["A", "B", "C"], "w": [1, 2, 1], "noise": [True, False, True]},
        schema={"v": pl.String, "w": pl.Int32, "noise": pl.Boolean},
    )
    penalty = pl.DataFrame(
        {"v": ["B", "C"], "w": [2, 1], "moq": [1.5, 2.5]},
        schema={"v": pl.String, "w": pl.Int64, "moq": pl.Float64},
    )
    rows = pl.DataFrame(
        {"v": ["A", "B", "C"], "w": [1, 2, 1], "min_quantity": [4.0, 3.0, 2.0]},
        schema={"v": pl.String, "w": pl.Int64, "min_quantity": pl.Float64},
    )

    with (
        gp.Env(params={"OutputFlag": 0}) as env,
        gp.Model(env=env) as ref,
        gp.Model(env=env) as new,
    ):
        d_ref = polarobi.GurobiFrame(x_dist).add_vars(ref, by=keys, name="d")
        nd_ref = polarobi.GurobiFrame(x_nd).add_vars(ref, by=keys, name="nd")
        b_ref = polarobi.GurobiFrame(breach).add_vars(ref, by=keys, name="b")
        p_ref = polarobi.GurobiFrame(penalty).add_vars(ref, by=keys, name="p", idx="p_idx")

        d_new = polarobi.GurobiFrame(x_dist).add_vars(new, by=keys, name="d")
        nd_new = polarobi.GurobiFrame(x_nd).add_vars(new, by=keys, name="nd")
        b_new = polarobi.GurobiFrame(breach).add_vars(new, by=keys, name="b")
        p_new = polarobi.GurobiFrame(penalty).add_vars(new, by=keys, name="p", idx="p_idx")

        # Repeated variable within and across sources: d_new appears again with a Series coef.
        series_coef = pl.Series([0.5, 1.5])
        terms = pl.concat(
            [
                d_ref.select(*keys, "idx", coef=pl.col("quantity").cast(pl.Float64)),
                nd_ref.select(*keys, "idx", coef=pl.col("quantity").cast(pl.Float64)),
                b_ref.select(*keys, "idx", coef=pl.lit(1.0)),
                p_ref.select(
                    *keys, pl.col("p_idx").alias("idx"), coef=-pl.col("moq").cast(pl.Float64)
                ),
                d_ref.select(*keys, "idx", coef=series_coef),
            ],
            how="vertical_relaxed",
        )
        got_ref = polarobi.GurobiFrame(terms).add_constrs(
            ref, rows, by=keys, sense=">=", rhs="min_quantity", coef="coef", name="cap", idx="c_idx"
        )
        got_new = polarobi.add_constrs(
            new,
            [
                (d_new, "quantity"),
                (nd_new, "quantity"),
                b_new,
                (p_new, -pl.col("moq"), "p_idx"),
                (d_new, series_coef),
            ],
            rows,
            by=keys,
            sense=">=",
            rhs="min_quantity",
            name="cap",
            idx="c_idx",
        )
        ref.update()
        new.update()
        assert got_new.equals(got_ref)
        assert got_new.schema["w"] == pl.Int64
        assert new.Fingerprint == ref.Fingerprint

        # Without explicit rows, key dtypes unify to Int64 across sources.
        no_rows = polarobi.add_constrs(
            new, [(d_new, "quantity"), (nd_new, "quantity")], by=keys, sense=">="
        )
        assert no_rows.schema["w"] == pl.Int64
