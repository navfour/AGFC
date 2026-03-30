"""Central place for configuration objects/enums used by dynamic graph runs.

Intended usage:
    * Define dataclasses/pydantic models describing snapshot inputs and modes.
    * Enumerate adjacency pipelines (static, snapshot, stacked, multi-adaptive).
    * Provide helpers for translating CLI flags into structured configs.
By separating config from runtime logic we can keep `dynamic_run_snapshot.py`
lean and make the experiment surface area easier to reason about.
"""
