import os
import sys
from pathlib import Path

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)
root_directory = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_directory))
grandparent = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(grandparent))
great_grandparent = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(great_grandparent))

from config.benchmark.tpch_string_keys import get_tpch_benchmark_str_keys
from config.systems.duckdb import Unified_String_Dictionary, USSR_SALT_CLEAN_FLAT_VEC_JOIN_NEW, USSR_SALT_CLEAN_FLAT_VEC_JOIN, USSR_SALT_CLEAN, UnifiedStringDictionary_lock_free_512K, DUCK_DB_USSR, DUCK_DB_MAIN, DUCK_DB_USSR_predicate_materialize_once, DUCK_DB_USSR_no_singleton, DUCK_DB_USSR_stable_version
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    sfs = [1, 10]
    for sf in sfs:
        config: RunConfig = {
            'name': f'USSR_vs_baseline_tpch_sf{sf}',
            'run_settings': {
                'n_parallel': 1,
                'n_runs': 6,
            },
            'system_settings': [
                # {'n_threads': 1},
                # {'n_threads': 2},
                # {'n_threads': 4},
                {'n_threads': 8},
            ],
            'systems': [DUCK_DB_MAIN, Unified_String_Dictionary],
            'benchmarks': get_tpch_benchmark_str_keys([sf]),  # Pass as a single-element list
        }
        run(config)


if __name__ == "__main__":
    main()