import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.microbenchmark_strings_variable_length import get_string_len_benchmark
from config.systems.duckdb import DUCK_DB_USSR, DUCK_DB_MAIN
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    sfs = [8, 16, 32, 64, 128, 256, 512]
    config: RunConfig = {
        'name': 'USSR_vs_baseline_microbenchmark_variable_length',
        'run_settings': {
            'n_parallel': 5,
            'n_runs': 5,
        },
        'system_settings': [
            {'n_threads': 1},
            # {'n_threads': 2},
            # {'n_threads': 4},
            # {'n_threads': 8},
        ],
        'systems': [DUCK_DB_USSR, DUCK_DB_MAIN],
        'benchmarks': get_string_len_benchmark(sfs),
    }
    run(config)


if __name__ == "__main__":
    main()