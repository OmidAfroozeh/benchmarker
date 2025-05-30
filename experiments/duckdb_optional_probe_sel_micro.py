import os
import sys

from config.benchmark.join_micro_probe import get_join_micro_probe_benchmark

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.tpch import get_tpch_benchmark
from config.systems.duckdb import DUCK_DB_NIGHTLY_BUILD_LOCALLY, DUCK_DB_OPTIONAL_PROBE_SEL
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():

    config: RunConfig = {
        'name': 'duckdb_optional_probe_sel_experiment_micro',
        'run_settings': {
            'n_parallel': 1,
            'n_runs': 5,

        },
        'system_settings': [
            {'n_threads': 1},
            {'n_threads': 4},
            {'n_threads': 8},
        ],

        'systems': [DUCK_DB_NIGHTLY_BUILD_LOCALLY, DUCK_DB_OPTIONAL_PROBE_SEL],
        'benchmarks': [get_join_micro_probe_benchmark()],
    }
    run(config)

if __name__ == "__main__":
    main()