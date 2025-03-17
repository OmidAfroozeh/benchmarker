import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.public_bi_commongovernment import get_tpch_benchmark
from config.systems.duckdb import DUCK_DB_USSR, DUCK_DB_MAIN
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    config: RunConfig = {
        'name': 'USSR_vs_baseline',
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
        'benchmarks': get_publicbi_commongov(),
    }
    run(config)


if __name__ == "__main__":
    main()