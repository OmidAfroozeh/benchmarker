import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.tpcds import get_tpcds_benchmark
from config.systems.duckdb import DUCK_DB_USSR_no_singleton_new_api, DUCK_DB_MAIN
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    sfs = [1, 10, 30]
    for sf in sfs:
        config: RunConfig = {
            'name': f'USSR_vs_baseline_tpcds_sf{sf}',
            'run_settings': {
                'n_parallel': 5,
                'n_runs': 5,
            },
            'system_settings': [
                # {'n_threads': 1},
                # {'n_threads': 2},
                # {'n_threads': 4},
                {'n_threads': 8},
            ],
            'systems': [DUCK_DB_USSR_no_singleton_new_api, DUCK_DB_MAIN],
            'benchmarks': get_tpcds_benchmark([sf]),  # Pass as a single-element list
        }
        run(config)


if __name__ == "__main__":
    main()