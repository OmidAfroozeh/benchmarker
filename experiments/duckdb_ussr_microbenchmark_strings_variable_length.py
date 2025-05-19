import os
import sys

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

from config.benchmark.microbenchmark_strings_variable_length import get_string_len_benchmark
from config.systems.duckdb import ussr_2x_size, DUCK_DB_USSR_stable_version_operator_bttr_strs_local, DUCK_DB_USSR_stable_version_operator_bttr_strs_flattening, DUCK_DB_USSR_stable_version_operator_bttr_strs_col_seg, DUCK_DB_USSR_stable_version_operator, DUCK_DB_USSR_stable_version_2x, DUCK_DB_USSR_upper_limit, DUCK_DB_USSR_stable_version, DUCK_DB_USSR, DUCK_DB_MAIN, DUCK_DB_USSR_refactored, DUCK_DB_USSR_memcpy_optimization, DUCK_DB_USSR_new_lock, DUCK_DB_USSR_predicate_materialize_once, DUCK_DB_USSR_no_singleton, DUCK_DB_USSR_no_singleton_new_api
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    sfs = [8, 16, 32, 64, 128, 255, 512]
    config: RunConfig = {
        'name': 'USSR_vs_baseline_microbenchmark_variable_length',
        'run_settings': {
            'n_parallel': 1,
            'n_runs': 5,
        },
        'system_settings': [
            {'n_threads': 2},
            # {'n_threads': 2},
            # {'n_threads': 4},
            # {'n_threads': 8},
        ],
        'systems': [DUCK_DB_MAIN, ussr_2x_size],
        'benchmarks': get_string_len_benchmark(sfs),
    }
    run(config)


if __name__ == "__main__":
    main()