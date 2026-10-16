---
title: Benchmarks
description: How long polarobi takes to build a model compared to plain gurobipy, and how to reproduce it.
---

???+ tip "TL;DR"
Building a model with a million variables and their constraints took **1.6 s with polarobi** and **9.7 s with
the faster of two common gurobipy styles**, 6.0 times faster, with identical models. At ten million it was 22.4 s
against 113.4 s. The gap grows up to a million rows and holds at five to six times beyond; at a thousand rows, all
builds take under 30 ms.

## Build Time

<div style="text-align: center" markdown>

![Build time against model size, both axes linear: polarobi stays below both gurobipy builds from 10,000 rows on](img/benchmark-build-time-light.svg#only-light)
![Build time against model size, both axes linear: polarobi stays below both gurobipy builds from 10,000 rows on](img/benchmark-build-time-dark.svg#only-dark)

</div>
<div style="text-align: center" markdown>

|       Rows | gurobipy, `addLConstr` loop | gurobipy, `addConstrs` | polarobi | Speedup |
|-----------:|----------------------------:|-----------------------:|---------:|--------:|
|      1,000 |                       16 ms |                  26 ms |    17 ms |    1.0x |
|     10,000 |                       71 ms |                 108 ms |    26 ms |    2.7x |
|    100,000 |                      839 ms |                  1.0 s |   172 ms |    4.9x |
|  1,000,000 |                       9.7 s |                 11.4 s |    1.6 s |    6.0x |
| 10,000,000 |                     113.4 s |                131.2 s |   22.4 s |    5.1x |

</div>
The speedup is measured against the faster gurobipy build, which was the `addLConstr` loop at every size. Times are
the median of three runs after an untimed warm-up, for building the whole model including the final `update()`. All
three builds produced the same `Fingerprint`, and the same variable and constraint names, at every size. At a
thousand rows polarobi's fixed cost per call, mostly polars planning its joins, about evens out what it saves.

## Where the Time Goes

<div style="text-align: center" markdown>

![Time per phase at 10M rows: variables 34.6 s with the gurobipy loop, 29.2 s with addConstrs and 9.2 s with polarobi; constraints 78.8 s, 102.0 s and 13.3 s](img/benchmark-phases-light.svg#only-light)
![Time per phase at 10M rows: variables 34.6 s with the gurobipy loop, 29.2 s with addConstrs and 9.2 s with polarobi; constraints 78.8 s, 102.0 s and 13.3 s](img/benchmark-phases-dark.svg#only-dark)

</div>

Constraints are where gurobipy loses most: every term is a tuple lookup into a `tupledict` and an entry in a
`LinExpr`, and every constraint a separate Python object on its way to Gurobi. `addConstrs` adds a comparison
object per constraint and its own bookkeeping on top, which made it the slower of the two styles here. polarobi
builds the same constraints with joins in polars and hands each family to Gurobi in one call. For variables, all
builds create one gurobipy handle per variable, which is most of polarobi's remaining time; see
[How it Works](concepts/how_it_works.md#registering-handles-with-gurobipy).

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
`tupledict` every time and be far slower. Nothing is solved, so no license size limits apply.

## Caveats

- **Your mileage may vary.** Results will differ across hardware,
  models, data shapes, and system load—the ratios within a single run are more telling than the absolute times.
  There is no magic: expect polarobi to be faster where creating millions of small interim Python objects has a
  noticeable impact on build time, or when a lot of data crosses the Python-to-Gurobi boundary.
- **Model building only.** Solving takes the same time either way, since the models are identical.

## Reproducing

The benchmark lives in `benchmarks/` in the repository:

```shell
uv run python benchmarks/build_model.py --sizes 1000 10000 100000 1000000 10000000 --cpus 0,2,4,6,8,10
uv run --group bench python benchmarks/plot.py benchmarks/results/windows.json
```

The first command builds the model all three ways at each size, checks that the fingerprints match, and writes the
timings, machine, package versions and settings to `benchmarks/results/<system>.json`. Without `--sizes` it runs
10K to 10M rows; 10M rows need about 11 GB of free memory. `--repeats` and `--seed` change the run. The second
command draws the figures on this page from a result file and prints the table.

`--cpus` pins the run to the given logical CPUs. On CPUs with performance and efficiency cores, such as Intel's
12th generation and later, pick one thread per performance core: otherwise the operating system moves the run
between core types, and identical builds differ by a factor of two.
