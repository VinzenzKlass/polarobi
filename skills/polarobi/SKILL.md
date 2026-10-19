---
name: polarobi
description: Build Gurobi optimization models from polars DataFrames with polarobi (GurobiFrame / df.grb), with variables, constraints and results as frames instead of Python loops. Use this skill whenever code imports polarobi or `GurobiFrame`, calls `.grb.add_vars` / `add_constrs`, or the user wants to build, speed up or port a gurobipy model whose data lives in polars, including replacing `iter_rows()` + `addVars` / `quicksum` / `addLConstr` loops, building large MIP/LP models from frames, or reading solutions (X, Pi, Slack) back into polars. Use it even if the user only says "gurobi + polars" without naming polarobi.
---

# polarobi

polarobi adds variables and constraints to a regular `gurobipy.Model` straight from polars frames: one
C API call per family, no Python object per term. The model stays a plain gurobipy model, so anything
not covered here (objective sense, parameters, quadratic or general constraints, `setObjective`) is
done with gurobipy as usual, in any mix.

It is experimental and relies on private gurobipy internals. It supports one gurobipy minor version
per release, so pin gurobipy next to it. When porting an existing model, check the result
(see "Verifying a model" below).

## The mental model

Everything is a frame with an `idx` column:

- `add_vars` returns the input frame plus `idx`, the Gurobi column index of each row's variable.
- `add_constrs` is called on a **terms** frame (one row per variable in a constraint: key columns,
  `idx`, coefficient) and returns the **rows** (one per constraint) plus `idx`, the constraint's row index.
- `get` / `set` read and write attributes for every row via `idx`. They also work on any filtered or
  joined frame derived from one.

Lookups like `x[a, b]` become joins that bring `idx` along. `quicksum` / `LinExpr` become terms frames.
Adding two expressions becomes `pl.concat` of their terms. Loops become one call per family.

## Setup

```python
import gurobipy as gp
import polars as pl
from polarobi import (
    GurobiFrame as grb,
)  # typed form of df.grb; prefer it, type checkers can't see df.grb

model = gp.Model()
```

`grb(df).add_vars(...)` and `df.grb.add_vars(...)` are identical. In method chains use `.pipe(grb)`.

## Variables

```python
x = grb(shipments).add_vars(
    model, by=["order", "week"], lb=0.0, ub=1.0, obj="cost", vtype="B", name="ship"
)
# names: ship[o1,2026-01] ... exactly like Model.addVars
```

- `by`: key columns for names and the duplicate check. Defaults to **all columns**, so pass it
  whenever the frame has data columns. Duplicate keys raise `KeyError` before anything is added.
  `by=[]` means no keys.
- `lb`, `ub`, `obj` accept a scalar, a column name, a `pl.Expr`, or a `pl.Series` (one value per row).
  Nulls are rejected. Use `float("inf")` for infinite bounds.
- `vtype`: one of `"C" "B" "I" "S" "N"` (or `gp.GRB.BINARY` etc.) for the whole family.
- `name`: a string gives `name[k1,k2]`; a `pl.Expr` gives free-form names (a null leaves the element unnamed).
  Omit it for huge models to save time and memory. Datetime keys can't be formatted into names; use an
  expression. Prefer `pl.concat_str` over `pl.format` (which can panic in polars 2.0.0).
- `idx="..."` renames the index column, which helps when one frame carries several families.
- `polarobi.add_var(model, *, lb=0.0, ub=float("inf"), obj=0.0, vtype="C", name=None, idx="idx")` adds a
  single variable and returns a one-row frame with `idx` (`name` used as is, without brackets), so it can
  appear in terms frames without calling `model.update()`.

## Constraints

Each group of terms with the same key becomes `sum(coef * x[var]) <sense> rhs`.

```python
# One constraint per order, coef 1 for each term:
grb(x).add_constrs(model, by="order", sense="<=", rhs=1.0, name="once")

# Per-constraint rhs from a rows frame; coef from a column of the terms:
grb(x).add_constrs(
    model, capacity, by="factory", sense="<=", rhs="capacity", coef="quantity", name="cap"
)
```

Key points:

- `coef` defaults to `1.0`. A column called `coef` is **not** used automatically; pass `coef="coef"`.
- `var` names the variable-index column in the terms (default `"idx"`).
- `rows` (2nd positional arg) is optional. Without it there is one constraint per distinct key, in order
  of first appearance, and `sense` / `rhs` / `name` must be constant or computable from the keys. With it,
  `rhs`, `sense` and `name` expressions and column names are evaluated **against the rows frame**.
  A `pl.Series` for `rhs` lines up with the rows frame as passed.
- Rows with no matching terms are dropped (`drop_empty=True`). Terms with no matching row are ignored.
  Repeated keys in rows give one constraint per row.
- `sense`: `"<" "<=" ">" ">=" "=" "=="`, or a `pl.Expr` / Series of those for mixed senses.
- `by=[]` sums all terms into **one** constraint (no rows frame allowed, and `name` is the plain name).
- A variable repeated within one constraint has its coefficients summed.

Common shapes:

```python
# Terms from a join (the "lookup")
terms = x.join(tooling, on="product").select("tooling", "week", "idx", coef=pl.col("hours"))
grb(terms).add_constrs(
    model, tool_cap, by=["tooling", "week"], sense="<=", rhs="capacity", coef="coef"
)

# Expression spanning several families: module-level add_constrs takes a list of sources
# Each source is frame, (frame, coef), or (frame, coef, var)
polarobi.add_constrs(
    model,
    [(x_a, "qty"), (x_b, "qty")],
    by="factory",
    sense="<=",
    rhs=100.0,
)

# Variable on the right-hand side moves left with a negated coef: sum(q*x) >= moq*y  ->  sum(q*x) - moq*y >= 0
polarobi.add_constrs(
    model,
    [(x, "quantity"), (y, -pl.col("moq"))],
    by="group",
    sense=">=",
    rhs=0.0,
    name="moq",
)

# Big-M linking x <= M*y per site: same pattern, y with coef=-M
```

Use `how="vertical_relaxed"` in `concat` so differing dtypes (for example a literal `coef=1.0` next to an
`i64` column) unify instead of failing.

## Results and attributes

```python
model.optimize()
sol = grb(x).get(model, "X").filter(pl.col("X") > 0.5)  # adds column "X"
duals = grb(cap_rows).get(model, "Pi")  # constraints work the same
prio = grb(x).get(model, "BranchPriority", dtype=pl.Int32)  # int attributes need dtype

grb(x.filter(~pl.col("allowed"))).set(model, "UB", 0.0)  # scalar, column name or Series
grb(y).set(model, "BranchPriority", 10)  # ints go through the int API
```

- Reading attributes of elements added since the last `update()` / `optimize()` raises `GurobiError`
  (as in gurobipy). Call `model.update()` first.
- `set` takes effect at the next update or optimize.
- Results are frames: join them back to your data and aggregate with polars. Don't loop over handles.

## Mixing with gurobipy

```python
vars_ = grb(x).handles(model)  # list[gp.Var], row order
td = grb(x).tupledict(model, ["order", "week"])  # gp.tupledict keyed like addVars
model.addLConstr(td["o1", "2026-01"] + td["o1", "2026-02"] <= 1)

extra = polarobi.add_var(model, name="extra")
terms = pl.concat(
    [x.select("idx", coef=pl.col("q")), extra.with_columns(coef=pl.lit(-1.0))],
    how="vertical_relaxed",
)
```

Removing elements with `model.remove(...)` shifts the indices of later elements after the next update,
and old `idx` columns then point at the wrong elements. If the model has pending deletions, any polarobi
call runs `model.update()` first to sync indices (otherwise polarobi leaves the model lazy). After
removals, re-read indices (for example `[v.index for v in grb(x).handles(model)]`) before reusing the
frames.

## Determinism

The model follows frame order. `group_by` and `unique` in polars do **not** keep order by default.
When terms or rows come out of them, pass `maintain_order=True` (or sort), or the model and its
`Fingerprint` change from run to run.

## Verifying a model

When porting from gurobipy, or whenever correctness matters, build both versions and compare:

```python
for m in (old, new):
    m.update()
assert (old.NumVars, old.NumConstrs, old.NumNZs) == (new.NumVars, new.NumConstrs, new.NumNZs)
assert old.Fingerprint == new.Fingerprint
```

The fingerprint depends on element order, so the original has to be deterministic too. Constraint names
follow `addConstrs` style (`min[V1,20260101]`). If the old code used other names, pass a name expression
to match them. `model.write("m.lp")` on a tiny instance is the quickest way to eyeball a formulation.

For a full gurobipy → polarobi translation table and a worked port, read
[references/porting.md](references/porting.md).
