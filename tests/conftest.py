"""
Hypothesis profiles, selected with the `HYPOTHESIS_PROFILE` environment variable:

- `dev` (default): quick enough for every local test run.
- `ci`: ten times the examples, for CI.
- `thorough`: a long search, for changes to the native code or the namespace semantics.
"""

import os

from hypothesis import HealthCheck, settings

_COMMON = settings(
    deadline=None,  # Gurobi calls vary in duration, timing is not what is tested.
    stateful_step_count=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,  # Failures print a blob to reproduce them with @reproduce_failure.
)
settings.register_profile("dev", _COMMON, max_examples=60)
settings.register_profile("ci", _COMMON, max_examples=600)
settings.register_profile("thorough", _COMMON, max_examples=6000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
