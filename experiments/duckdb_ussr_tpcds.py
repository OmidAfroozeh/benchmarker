import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.tpcds import get_tpcds_benchmark
from config.systems.duckdb import DUCK_DB_FIRST_TEST, DUCK_DB_NIGHTLY_BUILD_LOCALLY
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    sfs = [10, 30, 100, 300]
    config: RunConfig = {
        'name': 'USSR_vs_baseline_tpcds',
        'run_settings': {
            'n_parallel': 1,
            'n_runs': 3,
        },
        'system_settings': [
            {'n_threads': 8},
            # {'n_threads': 2},
            # {'n_threads': 4},
            # {'n_threads': 8},
        ],
        'systems': [DUCK_DB_USSR, DUCK_DB_MAIN],
        'benchmarks': get_tpcds_benchmark(sfs),
    }
    run(config)


if __name__ == "__main__":
    main()