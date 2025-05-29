#!/usr/bin/env python3
"""
Micro-benchmark driver (no argparse).

Edit STRING_LEN below to pick the fixed width of the generated strings.
"""

import os
import sys

# ── configuration knob ───────────────────────────────────────────────────────
STRING_LEN = 16          # ← tweak this and rerun

# ── project root on the import path ──────────────────────────────────────────
root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.microbenchmark_strings_variable_group_sizes import (
    get_string_len_benchmark,
)
from config.systems.duckdb import (
    UnifiedStringDictionary_16MB_dev_with_clientcontext, UnifiedStringDictionary_lock_free_16mB, UnifiedStringDictionary_lock_free_512K,
    DUCK_DB_MAIN,
)
from src.models import RunConfig
from src.runner.experiment_runner import run


def main() -> None:
    # group sizes (cardinalities) used in the benchmark
    # group_sizes = [100, 500, 1000, 3000, 5000, 10_000]
    # group_sizes = [100, 10_000, 20_000, 40_000, 80_000, 160_000, 320_000]
    group_sizes = [100, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
    # group_sizes = [100, 1000, 3000, 5000, 10000, 11000]


    config: RunConfig = {
        "name": f"USSR_vs_baseline_microbenchmark_variable_grp_sizes_len{STRING_LEN}",
        "run_settings": {
            "n_parallel": 1,
            "n_runs": 6,
        },
        "system_settings": [
            {'n_threads': 8},
            # {'n_threads': 1},
            # {'n_threads': 4},
            # {'n_threads': 8},
        ],
        "systems": [DUCK_DB_MAIN, UnifiedStringDictionary_lock_free_16mB],
        "benchmarks": get_string_len_benchmark(
            group_sizes, string_len=STRING_LEN
        ),
    }

    run(config)


if __name__ == "__main__":
    main()
