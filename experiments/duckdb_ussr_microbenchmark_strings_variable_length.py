import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.microbenchmark_strings_variable_length import get_string_len_benchmark
from config.systems.duckdb import UnifiedStringDictionary_lock_free_16mB, UnifiedStringDictionary_lock_free_512K, UnifiedStringDictionary_16MB_dev_with_clientcontext, UnifiedStringDictionary_1MB, UnifiedStringDictionary_2MB, UnifiedStringDictionary_4MB, UnifiedStringDictionary_8MB, UnifiedStringDictionary_16MB, DUCK_DB_MAIN
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    sfs = [8, 16, 32, 64, 128, 255]
    config: RunConfig = {
        'name': 'USSR_vs_baseline_microbenchmark_variable_length',
        'run_settings': {
            'n_parallel': 1,
            'n_runs': 6,
        },
        'system_settings': [
            {'n_threads': 8},
            # {'n_threads': 1},
            # {'n_threads': 4},
            # {'n_threads': 8},
        ],
        'systems': [DUCK_DB_MAIN, UnifiedStringDictionary_lock_free_16mB, UnifiedStringDictionary_16MB_dev_with_clientcontext],
        'benchmarks': get_string_len_benchmark(sfs),
    }
    run(config)


if __name__ == "__main__":
    main()