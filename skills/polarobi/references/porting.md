# Porting a gurobipy model to polarobi

## An Example

Demand rows are assigned to vendors per week. Each vendor week has a minimum quantity, which may be breached at a
penalty, and a maximum quantity. Vendor weeks without any demand get no constraints.

### Setup

```python
import gurobipy as gp
import polars as pl
from polarobi import GurobiFrame as grb

demand = pl.DataFrame(
    {
        "order": ["o1", "o2", "o3"],
        "vendor_code": ["V1", "V1", "V2"],
        "ex_go_date": ["20260101", "20260101", "20260101"],
        "quantity": [30.0, 45.0, 20.0],
    }
)
capacity = pl.DataFrame(
    {
        "vendor_code": ["V1", "V2", "V3"],
        "ex_go_date": ["20260101", "20260101", "20260101"],
        "min_qty": [50.0, 25.0, 10.0],
        "max_qty": [100.0, 80.0, 60.0],
    }
)
model = gp.Model()
```

### gurobipy

```python
keys = ("vendor_code", "ex_go_date")
x = model.addVars(demand.select("order", *keys).iter_rows(), vtype=gp.GRB.BINARY, name="x")
breach = model.addVars(capacity.select(*keys).iter_rows(), name="breach")

exprs = {}
for v, w, orders, quantities in demand.group_by(keys).agg("order", "quantity").iter_rows():
    exprs[v, w] = gp.LinExpr(quantities, [x[o, v, w] for o in orders])

for v, w, min_qty, max_qty in capacity.select(*keys, "min_qty", "max_qty").iter_rows():
    if (expr := exprs.get((v, w))) is None:
        continue
    model.addLConstr(expr + breach[v, w] >= min_qty, name=f"min[{v},{w}]")
    model.addLConstr(expr <= max_qty, name=f"max[{v},{w}]")
```

### polarobi

```python
from polarobi import add_constrs

keys = ("vendor_code", "ex_go_date")
x = grb(demand).add_vars(model, by=["order", *keys], vtype="B", name="x")
breach = grb(capacity).add_vars(model, by=keys, name="breach")

rows = capacity.join(x.select(keys), on=keys, how="semi")
add_constrs(model, [(x, "quantity"), breach], rows, by=keys, sense=">=", rhs="min_qty")
grb(x).add_constrs(model, rows, by=keys, sense="<=", rhs="max_qty", coef="quantity")
```

What changed:

- **Variables** come from the frames directly. `by` names the key columns, and names are formatted from them as
  `addVars` formats its keys.
- **Lookups** like `x[o, v, w]` disappear. `x` already carries each variable's `idx` next to its keys, so selecting
  columns is enough. Where variables and data live in different frames, a join on the keys does the lookup.
- **Expressions** become terms frames. Adding the breach variable to the quantity expression is a `concat` of the
  two term frames.
- **Skipping groups** with `continue` becomes filtering the rows frame. The maximum constraint does not even need
  that: rows without terms are dropped by default. The minimum constraint does, because the breach term would
  otherwise make every vendor week non-empty.
- **The loop** of `addLConstr` calls becomes one `add_constrs` call per family.

The names differ slightly, as polarobi names constraints the way `addConstrs` does: `min[V1,20260101]`. If you
depend on a particular name format, pass an expression, such as
`name=pl.concat_str(pl.lit("min_"), "vendor_code", pl.lit("_"), "ex_go_date")`.

## Pattern by Pattern

| gurobipy                                           | polarobi                                                             |
|----------------------------------------------------|----------------------------------------------------------------------|
| `model.addVars(df.select(keys).iter_rows(), ...)`  | `grb(df).add_vars(model, by=keys, ...)`                              |
| `model.addVar(ub=b, obj=c, name="v")`              | `add_var(model, ub=b, obj=c, name="v")`                              |
| `ub=list(caps.values())`                           | `ub="cap"`, a column of the same frame                               |
| `x[(a, b)]` lookups                                | A join on `a, b` that brings `idx` along                             |
| `gp.LinExpr(coefs, [x[k] for k in keys])`          | A terms frame with `idx` and a coefficient column, passed as `coef=` |
| `quicksum(x[k] for k in keys)`                     | A terms frame with `idx`; `coef` defaults to 1                       |
| `quicksum(e1 + e2)` across families                | `add_constrs(model, [(df1, c1), (df2, c2)], ...)`                    |
| `expr <= big_m * y`                                | `y`'s terms with `coef=-big_m`, `rhs=0`                              |
| `if not exprs: continue`                           | Default `drop_empty=True`, or a filtered rows frame                  |
| `for row in df.iter_rows(): model.addLConstr(...)` | `grb(terms).add_constrs(model, df, by=keys, ...)`                    |
| `y.setAttr("ub", 0)` for some groups               | `grb(y.filter(...)).set(model, "UB", 0.0)`                           |
| `setAttr("BranchPriority", list(y.values()), ...)` | `grb(y).set(model, "BranchPriority", 10)`                            |
| `[(k, v.X) for k, v in x.items() if v.X > 0]`      | `grb(x).get(model, "X").filter(pl.col("X") > 0)`                     |

## Checking the Port

Two models built from the same data should be identical, not just equally good. Compare them before switching:

```python
for m in (old_model, new_model):
    m.update()

assert old_model.NumVars == new_model.NumVars
assert old_model.NumConstrs == new_model.NumConstrs
assert old_model.NumNZs == new_model.NumNZs
assert old_model.Fingerprint == new_model.Fingerprint
```

`Fingerprint` depends on the order of variables and constraints. If the original model builds its constraints from
`group_by` or `unique` without `maintain_order=True`, its order changes from run to run, and so does its
fingerprint. Make the original deterministic first, then compare.
