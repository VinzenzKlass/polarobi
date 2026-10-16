---
title: Benchmarks
description: How long polarobi takes to build a model compared to gurobipy and gurobipy-pandas, and how to reproduce it.
---

???+ tip "TL;DR"
    Building a model with a million variables and their constraints took **1.3 s with polarobi**, **7.2 s with the
    faster of two common gurobipy styles** and 18.9 s with gurobipy-pandas, all with identical models: 5.7 times
    faster than gurobipy. The gap grows to ten times at 100,000 rows in this run; at a thousand rows, all builds take
    under 35 ms.

## Build Time

<div style="text-align: center" markdown>

![Build time against model size, both axes linear, with shaded bands from the fastest to the slowest run: gurobipy-pandas rises steepest to 18.9 s at 1M rows, the two plain gurobipy builds follow at 8.6 s and 7.2 s, and polarobi stays close to the axis at 1.3 s](img/benchmark-build-time-light.svg#only-light)
![Build time against model size, both axes linear, with shaded bands from the fastest to the slowest run: gurobipy-pandas rises steepest to 18.9 s at 1M rows, the two plain gurobipy builds follow at 8.6 s and 7.2 s, and polarobi stays close to the axis at 1.3 s](img/benchmark-build-time-dark.svg#only-dark)

</div>

<div class="benchmark-table">
<table>
<thead><tr><th>Rows</th><th>gurobipy, <code>addLConstr</code> loop</th><th>gurobipy, <code>addConstrs</code></th><th>gurobipy-pandas</th><th>polarobi</th><th>Speedup</th></tr></thead>
<tbody>
<tr class="summary"><td><label title="Show each run"><input type="checkbox" aria-label="Show each run at 1,000 rows">1,000</label></td><td>16 ms</td><td>27 ms</td><td>33 ms</td><td>17 ms</td><td>1.0x</td></tr>
<tr class="run"><td>run 1</td><td>16 ms</td><td>27 ms</td><td>33 ms</td><td>17 ms</td><td></td></tr>
<tr class="run"><td>run 2</td><td>16 ms</td><td>27 ms</td><td>33 ms</td><td>17 ms</td><td></td></tr>
<tr class="run"><td>run 3</td><td>16 ms</td><td>28 ms</td><td>35 ms</td><td>16 ms</td><td></td></tr>
</tbody>
<tbody>
<tr class="summary"><td><label title="Show each run"><input type="checkbox" aria-label="Show each run at 10,000 rows">10,000</label></td><td>125 ms</td><td>194 ms</td><td>188 ms</td><td>33 ms</td><td>3.8x</td></tr>
<tr class="run"><td>run 1</td><td>125 ms</td><td>207 ms</td><td>188 ms</td><td>31 ms</td><td></td></tr>
<tr class="run"><td>run 2</td><td>122 ms</td><td>190 ms</td><td>188 ms</td><td>36 ms</td><td></td></tr>
<tr class="run"><td>run 3</td><td>127 ms</td><td>194 ms</td><td>189 ms</td><td>33 ms</td><td></td></tr>
</tbody>
<tbody>
<tr class="summary"><td><label title="Show each run"><input type="checkbox" aria-label="Show each run at 100,000 rows">100,000</label></td><td>975 ms</td><td>1.2 s</td><td>1.2 s</td><td>98 ms</td><td>10.0x</td></tr>
<tr class="run"><td>run 1</td><td>954 ms</td><td>1.2 s</td><td>1.2 s</td><td>110 ms</td><td></td></tr>
<tr class="run"><td>run 2</td><td>978 ms</td><td>1.7 s</td><td>1.2 s</td><td>98 ms</td><td></td></tr>
<tr class="run"><td>run 3</td><td>975 ms</td><td>1.2 s</td><td>1.2 s</td><td>88 ms</td><td></td></tr>
</tbody>
<tbody>
<tr class="summary"><td><label title="Show each run"><input type="checkbox" aria-label="Show each run at 1,000,000 rows">1,000,000</label></td><td>7.2 s</td><td>8.6 s</td><td>18.9 s</td><td>1.3 s</td><td>5.7x</td></tr>
<tr class="run"><td>run 1</td><td>9.3 s</td><td>8.9 s</td><td>21.3 s</td><td>1.2 s</td><td></td></tr>
<tr class="run"><td>run 2</td><td>7.2 s</td><td>8.6 s</td><td>18.9 s</td><td>1.3 s</td><td></td></tr>
<tr class="run"><td>run 3</td><td>6.9 s</td><td>8.4 s</td><td>17.9 s</td><td>1.6 s</td><td></td></tr>
</tbody>
</table>
</div>

Times are the median of three runs after an untimed warm-up, for building the whole model including the final
`update()`; expand a size to see each run. The speedup divides the median of the faster plain gurobipy build, which
was the `addLConstr` loop at every size, by polarobi's. All four builds produced the same `Fingerprint`, and the
same variable and constraint names, at every size. At a thousand rows polarobi's fixed cost per call, mostly polars
planning its queries, about evens out what it saves.

## Where the Time Goes

<div style="text-align: center" markdown>

![Time per phase at 1M rows: variables 2.2 s with the gurobipy loop, 2.1 s with addConstrs, 3.9 s with gurobipy-pandas and 0.7 s with polarobi; constraints 5.0 s, 6.5 s, 15.0 s and 0.5 s](img/benchmark-phases-light.svg#only-light)
![Time per phase at 1M rows: variables 2.2 s with the gurobipy loop, 2.1 s with addConstrs, 3.9 s with gurobipy-pandas and 0.7 s with polarobi; constraints 5.0 s, 6.5 s, 15.0 s and 0.5 s](img/benchmark-phases-dark.svg#only-dark)

</div>

Constraints are where the other builds lose most: every term is an entry in a `LinExpr`, and every constraint a
separate Python object on its way to Gurobi. In the plain gurobipy builds each term is also a tuple lookup into a
`tupledict`; `addConstrs` adds a comparison object per constraint and its own bookkeeping on top. gurobipy-pandas
builds its expressions with pandas arithmetic over Series of `Var` objects and then adds one constraint per row,
which made it the slowest build at a million rows. polarobi groups the terms in polars and hands each family to
Gurobi in one call. For variables, all builds create one gurobipy handle per variable, which is most of polarobi's
remaining time; see [How it Works](concepts/how_it_works.md#registering-handles-with-gurobipy).

## The Model

A typical allocation model, generated from a fixed seed:

- `x`: one binary variable per option, an (order, factory, week) combination with a quantity and a value. The
  row count is the number of options, about four per order, over 50 factories and 52 weeks.
- `assign`: every order is chosen at most once.
- `capacity`: the quantity per factory and week stays within its capacity.
- `y`, `moq_min`, `moq_max`: a factory week either produces nothing or at least a minimum quantity, a big-M link
  through one binary variable per factory week.

The gurobipy builds are written the way such models usually are: `iter_rows()` into `addVars`, and
`group_by().agg()` with tupledict lookups into one `LinExpr` per key. They differ only in how they add the
constraints:

- **`addLConstr` loop:** one `addLConstr` call per constraint.
- **`addConstrs`:** one `addConstrs` call per family, over a generator of the keys.

Both build each expression once; the also common `x.sum(order, "*", "*")` per constraint would scan the whole
`tupledict` every time and be far slower.

A third build uses gurobipy-pandas, written as its documentation suggests: `gppd.add_vars` on frames indexed by
their keys, `groupby().sum()` into one `LinExpr` per key, and `gppd.add_constrs`. Its inputs are converted to pandas
before the clock starts. It adds variables with one `addMVar` call, but constraints one `addLConstr` call per row.

Nothing is solved, so no license size limits apply.

## Caveats

- **Your mileage may vary.** Results will differ across hardware,
  models, data shapes, and system load—the ratios within a single run are more telling than the absolute times.
  There is no magic: expect polarobi to be faster where creating millions of small interim Python objects has a
  noticeable impact on build time, or when a lot of data crosses the Python-to-Gurobi boundary.
- **Model building only.** Solving takes the same time either way, since the models are identical.

## Reproducing

The benchmark lives in `benchmarks/` in the repository:

```shell
uv run --group bench python benchmarks/build_model.py --sizes 1000 10000 100000 1000000 10000000
uv run --group bench python benchmarks/plot.py benchmarks/results/windows.json
```

The first command builds the model all four ways at each size, checks that the fingerprints and names match, and
writes the timings, machine, package versions and settings to `benchmarks/results/<system>.json`. Without
`--sizes` it runs 10K to 10M rows; 10M rows need well over 11 GB of free memory, since gurobipy-pandas holds a
pandas copy of the inputs. `--repeats` and `--seed` change the run. The second command draws the figures on this
page from a result file and prints the table.

`--cpus` pins the run to the given logical CPUs. On CPUs with performance and efficiency cores, such as Intel's
12th generation and later, pick one thread per performance core: otherwise the operating system moves the run
between core types, and identical builds differ by a factor of two.
