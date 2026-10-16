---
title: Equivalence with gurobipy
description: What polarobi guarantees about the models it builds, and how that is tested.
---

!!! tip "TL;DR"
    For every operation that gurobipy also offers, polarobi builds exactly the model gurobipy builds: the same
    variables, constraints, coefficients, names and order, down to signed zeros. Its test suite checks that against
    gurobipy itself on random inputs.

## The Guarantee

Take any sequence of model edits that can be written with gurobipy: adding variables with `addVars` or `addVar`,
constraints with `addConstrs` or `addLConstr`, setting attributes, removing elements and updating, in any mix with
plain gurobipy calls. Doing the same through polarobi produces a model that is identical to gurobipy's in every
respect Gurobi exposes:

- model attributes, including `NumNZs` and `Fingerprint`
- every variable's name, bounds, objective coefficient, type and branching priority
- every constraint's name, sense, right-hand side and coefficients
- the MPS file Gurobi writes for it, byte for byte
- gurobipy's own view of the model: `getVars()`, `getConstrs()` and every handle's `index`

Values read with `get` match `Model.getAttr`, and frames returned by polarobi match what the same operation returns
when written with gurobipy.

## gurobipy's Rules

Matching gurobipy exactly means following some rules that are easy to miss. polarobi implements each of them:

| Situation                                              | gurobipy, and polarobi                                 |
|--------------------------------------------------------|--------------------------------------------------------|
| Names from keys                                        | `str()` of each key: `True`, `1e-07`, `None` for nulls |
| A variable repeated within a constraint                | Coefficients summed left to right, see below           |
| A coefficient that is zero or tiny, even after summing | Dropped by Gurobi                                      |
| A right-hand side of `-0.0`                            | Becomes `0.0`                                          |
| A bound of `-0.0`                                      | Kept as `-0.0`                                         |
| A bound or other value at Gurobi's infinity            | Reads as `inf`                                         |
| Repeated keys in the rows of a constraint family       | One constraint per row, with the same name             |
| Repeated keys in a variable family                     | `KeyError`                                             |
| Reading attributes of elements awaiting `update()`     | `GurobiError`                                          |

## Where polarobi Differs

- **Failed calls change nothing.** polarobi validates its input before its single call into Gurobi and registers
  handles only after it succeeded. gurobipy, by contrast, raises for duplicate keys in `addVars` only after adding
  the variables.
- **Repeated variables are summed left to right.** gurobipy sums a variable that repeats within a constraint in
  the order of an unstable sort, which depends on the row's other terms and has no rule worth reproducing. Where a
  sum depends on its order, the last bit of the coefficient can differ; polarobi's order is fixed and the same on
  every platform. The test suite's reference implementation sums repeats left to right as well.
- **Inputs gurobipy cannot express** have their own rules: a Series of right-hand sides lines up with the rows frame
  as given, and `by=[]` sums all terms into a single constraint.

## How It Is Tested

The test suite contains a reference implementation of every polarobi operation, written with public gurobipy calls
only, the way a gurobipy user would write it. Wherever gurobipy has an equivalent, such as `addVars` and
`addConstrs`, gurobipy itself decides the names and errors.

A [Hypothesis](https://hypothesis.readthedocs.io/) state machine then drives two models through the same random
sequences of operations, one through polarobi and one through the reference, interleaved with plain gurobipy calls on
both and with removals left pending. The inputs vary in everything polarobi accepts:

- key columns of every supported dtype, with nulls, `-0.0`, separators and empty strings
- every input form: scalars, column names, expressions, literal expressions and Series
- frames split into several chunks, or sliced to a non-zero offset in their buffers
- variable index columns of different integer dtypes, repeated variables and repeated keys

Every return value must match, and whenever Hypothesis chooses to, and at the end of every sequence, the two models
must be identical in all of the respects listed above. A separate set of tests covers errors and edge cases, such as
duplicate keys, reading pending elements and invalid input.

Several of the rules in the table above were found this way: each was a difference between polarobi and gurobipy that
the state machine turned up and that polarobi now follows.

## Known Limitations

- **Non-ASCII keys in names**: gurobipy itself cannot read back the names its `addVars` writes for them, so they are
  not tested. Use expression names if you need them.
- **`pl.format` in polars 2.0.0** can panic on string columns that consist of several chunks and were filtered.
  `pl.concat_str` builds the same names without the problem.
