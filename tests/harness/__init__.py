"""
Equivalence harness: every `df.grb` call must build exactly what plain gurobipy builds.

- `specs`: one namespace call as data, applied identically to both sides.
- `native`: applies a spec through polarobi's `df.grb` (the only untyped module).
- `reference`: the specification. The documented behaviour, implemented with public gurobipy
  calls only, using gurobipy itself (`addVars`, `addConstrs`) wherever it has an equivalent.
- `compare.assert_identical`: model attributes and `Fingerprint`, every element attribute, the
  matrix row by row, the written MPS file, and gurobipy's handle bookkeeping.
- `strategies`: Hypothesis inputs, varying dtypes, input forms and memory layout.

`tests/test_equivalence.py` runs both sides through random operation sequences;
`tests/test_parity.py` covers errors and the cases those inputs exclude.

Adding or changing a namespace feature:

1. Write its behaviour into `reference` first, as a gurobipy user would write it.
2. Add the arguments to its spec in `specs`, the call in `native`, and inputs in `strategies`.
3. New operations get a rule in `test_equivalence.Equivalence`; error behaviour goes to
   `test_parity`.
4. Run `HYPOTHESIS_PROFILE=thorough uv run pytest tests/test_equivalence.py` before merging.

When the harness finds a difference, check gurobipy's behaviour directly before changing either
side: several rules here (signed zeros, summed repeats, name formatting) were found that way.
"""
