#!/usr/bin/env python3
"""
Run the “different USSR sizes” benchmark (Zipf 1.2 skew).

Tweak STRING_LEN below and rerun; no CLI parsing required.
"""

import os
import sys

# ── configuration knob ───────────────────────────────────────────────────────
STRING_LEN = 32        # ← edit this value to the desired fixed string width

# ── project root on the import path ──────────────────────────────────────────
root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.microbenchmark_ussr_size import get_string_len_benchmark
from config.systems.duckdb import (
    UnifiedStringDictionary_16MB_dev_with_clientcontext,
    DUCK_DB_MAIN, UnifiedStringDictionary_lock_free_16mB
)
from src.models import RunConfig
from src.runner.experiment_runner import run


def main() -> None:
    # cardinalities / group sizes
    group_sizes = [100, 10_000, 20_000, 40_000, 80_000, 160_000, 320_000]

    config: RunConfig = {
        "name": f"Different_USSR_sizes_len{STRING_LEN}",
        "run_settings": {"n_parallel": 1, "n_runs": 11},
        "system_settings": [{"n_threads": 8}],
        "systems": [DUCK_DB_MAIN, UnifiedStringDictionary_lock_free_16mB],
        "benchmarks": get_string_len_benchmark(
            group_sizes, string_len=STRING_LEN
        ),
    }
    run(config)


if __name__ == "__main__":
    main()
