---
title: Benchmarks
description: How long polarobi takes to build a model compared to gurobipy and gurobipy-pandas, and how to reproduce it.
---

???+ tip "TL;DR"
    Building a model with ten million variables and their constraints took **10.2 s with polarobi** and **90.8 s
    with the fastest alternative**, a plain gurobipy loop: 8.9 times faster, with identical models. At a million rows
    it was 0.9 s against 7.7 s. polarobi was the fastest build at every size, from 1.4 times faster at a thousand
    rows, and gurobipy-pandas the slowest from a million rows on.

## Build Time

<div style="text-align: center" markdown>

![Build time against model size, both axes linear, with shaded bands from the fastest to the slowest run: at 10M rows gurobipy-pandas takes 146.2 s, the two plain gurobipy builds 99.3 s and 90.8 s, and polarobi 10.2 s, close to the axis at every size](img/benchmark-build-time-light.svg#only-light)
![Build time against model size, both axes linear, with shaded bands from the fastest to the slowest run: at 10M rows gurobipy-pandas takes 146.2 s, the two plain gurobipy builds 99.3 s and 90.8 s, and polarobi 10.2 s, close to the axis at every size](img/benchmark-build-time-dark.svg#only-dark)

</div>

<div class="benchmark-table">
<table>
<thead><tr><th>Rows</th><th>gurobipy, <code>addLConstr</code> loop</th><th>gurobipy, <code>addConstrs</code></th><th>gurobipy-pandas</th><th>polarobi</th><th>Speedup</th></tr></thead>
<tbody>
<tr class="summary"><td><label title="Show each run"><input type="checkbox" aria-label="Show each run at 1,000 rows">1,000</label></td><td>13 ms</td><td>22 ms</td><td>27 ms</td><td>9 ms</td><td title="Against gurobipy, addLConstr loop">1.4x</td></tr>
<tr class="run"><td>run 1</td><td>13 ms</td><td>22 ms</td><td>26 ms</td><td>9 ms</td><td></td></tr>
<tr class="run"><td>run 2</td><td>13 ms</td><td>23 ms</td><td>27 ms</td><td>9 ms</td><td></td></tr>
<tr class="run"><td>run 3</td><td>13 ms</td><td>22 ms</td><td>27 ms</td><td>10 ms</td><td></td></tr>
</tbody>
<tbody>
<tr class="summary"><td><label title="Show each run"><input type="checkbox" aria-label="Show each run at 10,000 rows">10,000</label></td><td>69 ms</td><td>101 ms</td><td>92 ms</td><td>20 ms</td><td title="Against gurobipy, addLConstr loop">3.5x</td></tr>
<tr class="run"><td>run 1</td><td>70 ms</td><td>101 ms</td><td>92 ms</td><td>20 ms</td><td></td></tr>
<tr class="run"><td>run 2</td><td>69 ms</td><td>101 ms</td><td>93 ms</td><td>20 ms</td><td></td></tr>
<tr class="run"><td>run 3</td><td>67 ms</td><td>100 ms</td><td>91 ms</td><td>19 ms</td><td></td></tr>
</tbody>
<tbody>
<tr class="summary"><td><label title="Show each run"><input type="checkbox" aria-label="Show each run at 100,000 rows">100,000</label></td><td>519 ms</td><td>635 ms</td><td>626 ms</td><td>79 ms</td><td title="Against gurobipy, addLConstr loop">6.5x</td></tr>
<tr class="run"><td>run 1</td><td>518 ms</td><td>642 ms</td><td>629 ms</td><td>83 ms</td><td></td></tr>
<tr class="run"><td>run 2</td><td>525 ms</td><td>631 ms</td><td>623 ms</td><td>77 ms</td><td></td></tr>
<tr class="run"><td>run 3</td><td>514 ms</td><td>631 ms</td><td>624 ms</td><td>78 ms</td><td></td></tr>
</tbody>
<tbody>
<tr class="summary"><td><label title="Show each run"><input type="checkbox" aria-label="Show each run at 1,000,000 rows">1,000,000</label></td><td>7.7 s</td><td>8.5 s</td><td>11.0 s</td><td>882 ms</td><td title="Against gurobipy, addLConstr loop">8.8x</td></tr>
<tr class="run"><td>run 1</td><td>7.7 s</td><td>8.5 s</td><td>11.0 s</td><td>885 ms</td><td></td></tr>
<tr class="run"><td>run 2</td><td>7.7 s</td><td>8.5 s</td><td>11.0 s</td><td>879 ms</td><td></td></tr>
<tr class="run"><td>run 3</td><td>7.7 s</td><td>8.6 s</td><td>10.9 s</td><td>884 ms</td><td></td></tr>
</tbody>
<tbody>
<tr class="summary"><td><label title="Show each run"><input type="checkbox" aria-label="Show each run at 10,000,000 rows">10,000,000</label></td><td>90.8 s</td><td>99.3 s</td><td>146.2 s</td><td>10.2 s</td><td title="Against gurobipy, addLConstr loop">8.9x</td></tr>
<tr class="run"><td>run 1</td><td>90.5 s</td><td>99.0 s</td><td>146.0 s</td><td>10.2 s</td><td></td></tr>
<tr class="run"><td>run 2</td><td>91.2 s</td><td>99.6 s</td><td>145.9 s</td><td>10.2 s</td><td></td></tr>
<tr class="run"><td>run 3</td><td>90.7 s</td><td>99.3 s</td><td>146.6 s</td><td>10.2 s</td><td></td></tr>
</tbody>
</table>
</div>

Times are the mean of three runs after an untimed warm-up, for building the whole model including the final
`update()`; expand a size to see each run. The speedup divides the mean of the next fastest build by polarobi's, and
hovering it names that build: the gurobipy `addLConstr` loop at every size. All four builds produced the same
`Fingerprint`, and the same variable and constraint names, at every size.

## Where the Time Goes

<div style="text-align: center" markdown>

![Time per phase at 10M rows: variables 24.7 s with the gurobipy loop, 24.9 s with addConstrs, 24.9 s with gurobipy-pandas and 5.1 s with polarobi; constraints 66.0 s, 74.4 s, 121.1 s and 5.1 s](img/benchmark-phases-light.svg#only-light)
![Time per phase at 10M rows: variables 24.7 s with the gurobipy loop, 24.9 s with addConstrs, 24.9 s with gurobipy-pandas and 5.1 s with polarobi; constraints 66.0 s, 74.4 s, 121.1 s and 5.1 s](img/benchmark-phases-dark.svg#only-dark)

</div>

Constraints are where the other builds lose most: every term is an entry in a `LinExpr`, and every constraint a
separate Python object on its way to Gurobi. In the plain gurobipy builds each term is also a tuple lookup into a
`tupledict`; `addConstrs` adds a comparison object per constraint and its own bookkeeping on top. gurobipy-pandas
builds its expressions with pandas arithmetic over Series of `Var` objects and then adds one constraint per row,
which made it the slowest build from a million rows on. polarobi groups the terms in polars and hands each family to
Gurobi in one call. For variables, all builds create one gurobipy handle per variable, which is half of polarobi's
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

## The Machine

The results on this page come from a virtual machine on Google Cloud, so that anyone can rerun them on the same
hardware:

- **Machine:** `c4-highmem-16` in `europe-west1-c`, an Intel Granite Rapids CPU with one vCPU per core, limited to 4
  visible cores, and 124 GB of memory.
- **System:** Ubuntu 26.04 minimal (`ubuntu-minimal-2604-resolute-amd64`), Linux 7.0.0-1011-gcp.
- **Software:** Python 3.14.7, polarobi 0.1.4, polars 2.0.0, gurobipy 14.0.0b1, gurobipy-pandas 1.2.2 and pandas
  3.0.6.

All its cores are of one type, so the run needs no CPU pinning. From a million rows on, the three runs of every build
stayed within 1% of each other; below that, where a run takes milliseconds, within 8%.

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
uv run --group bench python benchmarks/plot.py benchmarks/results/results.json
```

The first command builds the model all four ways at each size, checks that the fingerprints and names match, and
writes the timings, machine, package versions and settings to `benchmarks/results/<system>.json`; the results on
this page are `../benchmarks/results/gcp-c4-highmem-16.json`. Without
`--sizes` it runs 10K to 10M rows; 10M rows need well over 11 GB of free memory, since gurobipy-pandas holds a
pandas copy of the inputs. `--repeats` and `--seed` change the run. The second command draws the figures on this
page from a result file and prints the table.

`--cpus` pins the run to the given logical CPUs. On CPUs with performance and efficiency cores, such as Intel's
12th generation and later, pick one thread per performance core: otherwise the operating system moves the run
between core types, and identical builds differ by a factor of two.
