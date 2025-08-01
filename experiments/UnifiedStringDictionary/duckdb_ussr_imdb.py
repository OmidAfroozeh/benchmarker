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

from config.benchmark.imdb import get_imdb_benchmark
from config.systems.duckdb import Unified_String_Dictionary, UnifiedStringDictionary_1GB_full_insertion, UnifiedStringDictionary_initial_benchmark_32MB_upper_limit_smarter_insertion,ussr_2x_size_analyze_all_vecs, ussr_16MB, ussr_2x_size, DUCKDB_emit_DICT, DUCK_DB_USSR_stable_version_operator_bttr_strs_local, DUCK_DB_USSR, DUCK_DB_MAIN, DUCK_DB_USSR_new_lock, DUCK_DB_USSR_no_singleton
from src.models import RunConfig
from src.runner.experiment_runner import run


def main():
    config: RunConfig = {
        'name': 'USSR_vs_baseline_imdb',
        'run_settings': {
            'n_parallel': 1,
            'n_runs': 6,
        },
        'system_settings': [
            {'n_threads': 8},
            # {'n_threads': 1},
            # {'n_threads': 4},
            # {'n_threads': 8},
        ],
        'systems': [DUCK_DB_MAIN, Unified_String_Dictionary],
        'benchmarks': get_imdb_benchmark(),
    }
    run(config)


if __name__ == "__main__":
    main()