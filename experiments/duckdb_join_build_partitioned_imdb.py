import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)


from config.benchmark.imdb import get_imdb_benchmark
from config.systems.duckdb import DUCK_DB_MAIN, ussr_2x_size
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():

    config: RunConfig = {
        'name': 'duckdb_join_build_partitioned_imdb',
        'run_settings': {
            'n_parallel': 1,
            'n_runs': 4,
        },
        'system_settings': [
            {'n_threads': 1},
            {'n_threads': 2},
            {'n_threads': 4},
            {'n_threads': 8},
        ],
        'systems': [DUCK_DB_MAIN, ussr_2x_size],
        'benchmarks': get_imdb_benchmark(),
    }
    run(config)

if __name__ == "__main__":
    main()