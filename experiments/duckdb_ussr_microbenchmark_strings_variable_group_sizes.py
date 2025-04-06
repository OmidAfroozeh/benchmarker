import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.microbenchmark_strings_variable_group_sizes import get_string_len_benchmark
from config.systems.duckdb import DUCK_DB_USSR, DUCK_DB_MAIN, DUCK_DB_USSR_no_singleton
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    sfs = [100, 500, 1000, 2000, 5000, 10000, 20000, 80000, 600000]
    config: RunConfig = {
        'name': 'USSR_vs_baseline_microbenchmark_variable_grp_sizes',
        'run_settings': {
            'n_parallel': 5,
            'n_runs': 10,
        },
        'system_settings': [
            # {'n_threads': 1},
            # {'n_threads': 2},
            {'n_threads': 4},
            # {'n_threads': 8},
        ],
        'systems': [DUCK_DB_MAIN, DUCK_DB_USSR_no_singleton],
        'benchmarks': get_string_len_benchmark(sfs),
    }
    run(config)


if __name__ == "__main__":
    main()