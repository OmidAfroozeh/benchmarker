import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.clickbench import get_clickbench
from config.systems.duckdb import DUCKDB_base2, DUCKDB_emit_DICT, DUCK_DB_USSR_stable_version_operator_bttr_strs_local, DUCK_DB_USSR, DUCK_DB_MAIN, DUCK_DB_USSR_new_lock, DUCK_DB_USSR_no_singleton
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    config: RunConfig = {
        'name': 'USSR_vs_baseline_clickbench',
        'run_settings': {
            'n_parallel': 1,
            'n_runs': 10,
        },
        'system_settings': [
            {'n_threads': 6},
            # {'n_threads': 1},
            # {'n_threads': 4},
            # {'n_threads': 8},
        ],
        'systems': [DUCK_DB_MAIN, DUCKDB_base2],
        'benchmarks': get_clickbench(),
    }
    run(config)


if __name__ == "__main__":
    main()