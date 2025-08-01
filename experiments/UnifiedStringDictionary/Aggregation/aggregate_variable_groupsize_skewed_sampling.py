from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Ensure repo root is importable when executed directly
# ---------------------------------------------------------------------------
root_directory = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_directory))
grandparent = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(grandparent))
great_grandparent = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(great_grandparent))

# ---------------------------------------------------------------------------
# Unified benchmark utilities
# ---------------------------------------------------------------------------
from config.benchmark.synthetic_benchmark import ColumnSpec, generate_string_benchmark

# ---------------------------------------------------------------------------
# System definitions & experiment runner
# ---------------------------------------------------------------------------
from config.systems.duckdb import (
    DUCK_DB_MAIN,
    UnifiedStringDictionary_initial_benchmark,
    UnifiedStringDictionary_initial_benchmark_32MB_upper_limit,
    UnifiedStringDictionary_initial_benchmark_64MB,
    UnifiedStringDictionary_initial_benchmark_32MB_upper_limit_smarter_insertion,
    UnifiedStringDictionary_1GB_full_insertion,
    USSR_SALT_CLEAN,
    Unified_String_Dictionary,
    unified_string_dictionary_256MB_no_constraint,
    UnifiedStringDictionary_16MB,
    unified_string_dictionary_256MB_with_constraint

)
from src.models import DataSet, Benchmark, RunConfig, Query
from src.runner.experiment_runner import run

# ---------------------------------------------------------------------------
# get_data_path helper (fallback implementation outside repo)
# ---------------------------------------------------------------------------

from src.utils import get_data_path  # type: ignore

# =============================================================================
# ░░ Parameter grids ░░
# =============================================================================
LengthSpec = Union[int, Tuple[int, int]]  # fixed or (min, max)

# default grids
LENGTH_SPECS: Sequence[LengthSpec] = [32]
TOTAL_ROWS_LIST: Sequence[int] = [10_000_000]
# N_UNIQUE_LIST: Sequence[int] = [100, 1000, 3000]
N_UNIQUE_LIST: Sequence[int] = [15000, 30000, 60000, 120000, 240000, 480000, 960000]
# N_UNIQUE_LIST: Sequence[int] = [10000, 30000, 50000, 100_000]
# N_UNIQUE_LIST: Sequence[int] = [15000, 30000, 60000, 120000, 240000]
# N_UNIQUE_LIST: Sequence[int] = [500, 1000, 1500, 2000]

# N_UNIQUE_LIST: Sequence[int] = [1_000_000]

# zipf parameters to pin
PIN_VAR = 'zipf_s'
PIN_VALUES: Sequence[float] = [1.0]
DEFAULT_S_VALUES: Sequence[float] = [0.0]

# ---------------------------------------------------------------------------
# Fixed generation knobs
# ---------------------------------------------------------------------------
CHUNK_ROWS: int = 2_000_000
PARQUET_CODEC: str = "zstd"
SEED_BASE: int = 999

# ----------------------------------------------------------------------------
# Custom query list
# ----------------------------------------------------------------------------
CUSTOM_QUERIES: List[Query] = [
    {
        "name": "double_column_groupby",
        "index": 0,
        "run_script": {"duckdb": "SELECT str1, str2 FROM varchars GROUP BY str1, str2"},
    },
    # {
    #     "name": "constant_double_column_groupby",
    #     "index": 1,
    #     "run_script": {"duckdb": "SELECT 1, str1 FROM varchars GROUP BY 1, str1"},
    # },
    # {
    #     "name": "double_column_groupby",
    #     "index": 2,
    #     "run_script": {"duckdb": "SELECT str1, str2 FROM varchars GROUP BY str1, str2 limit 1"},
    # },
    # {
    #     "name": "double_column_groupby",
    #     "index": 3,
    #     "run_script": {"duckdb": "SELECT str1, str2, str3 FROM varchars GROUP BY str1, str2, str3"},
    # },
]


# =============================================================================
# ░░ Dataset assembly ░░
# =============================================================================

def len_spec_to_key(spec: LengthSpec) -> str:
    if isinstance(spec, int):
        return str(spec)
    lo, hi = spec
    return f"{lo}-{hi}"


def build_db_path(len_spec: LengthSpec, n_unique: int, s_val: float) -> str:
    dist_dir = f"varchars_grp_size_zipf={s_val}_nrows={TOTAL_ROWS_LIST[0] / 1_000_000}M_uniques={n_unique}_len={len_spec_to_key(len_spec)}"
    fname = f"varchars-grp-size-{n_unique}.db"
    rel = os.path.join(dist_dir, fname)
    return get_data_path(rel)


def make_column_specs(spec: LengthSpec, n_unique: int, s_val: float) -> List[ColumnSpec]:
    return [
        ColumnSpec("str1", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),
        ColumnSpec("str2", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),
        ColumnSpec("str3", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),

    ]


def assemble_datasets(
        length_specs: Sequence[LengthSpec] = LENGTH_SPECS,
        total_rows_list: Sequence[int] = TOTAL_ROWS_LIST,
        n_unique_list: Sequence[int] = N_UNIQUE_LIST,
        s_values: Sequence[float] = DEFAULT_S_VALUES,
) -> List[DataSet]:
    datasets: List[DataSet] = []
    combo_iter = itertools.product(length_specs, total_rows_list, n_unique_list, s_values)

    for idx, (len_spec, rows, uniques, s_val) in enumerate(combo_iter, start=1):
        db_path = Path(build_db_path(len_spec, uniques, s_val))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if not db_path.exists():
            generate_string_benchmark(
                duckdb_path=db_path,
                total_rows=rows,
                column_specs=make_column_specs(len_spec, uniques, s_val),
                chunk_rows=CHUNK_ROWS,
                parquet_codec=PARQUET_CODEC,
                seed=SEED_BASE + idx,
            )

        setup_script = {
            "duckdb": (
                f"ATTACH '{db_path}' (READ_ONLY); USE '{db_path.stem}';"
            ),
        }

        name_key = len_spec_to_key(len_spec)
        datasets.append(
            {
                "name": f"len{name_key}_uni{uniques}_zipf{s_val}",
                "setup_script": setup_script,
                "config": {"string_length": len_spec, "Domain Cardinality": uniques},
            }
        )
    return datasets


# =============================================================================
# ░░ Benchmark builder ░░
# =============================================================================

def build_benchmark(s_values_list: Sequence[float] = DEFAULT_S_VALUES) -> Benchmark:
    return {
        "name": f"string_benchmark_{PIN_VAR}",
        "datasets": assemble_datasets(s_values=s_values_list),
        "queries": CUSTOM_QUERIES,
    }


# =============================================================================
# ░░ Main entry-point ░░
# =============================================================================
RUN_SETTINGS = {"n_parallel": 1, "n_runs": 6}
SYSTEM_SETTINGS = [{"n_threads": 8}]
# SYSTEMS = [DUCK_DB_MAIN, Unified_String_Dictionary]
SYSTEMS = [DUCK_DB_MAIN, UnifiedStringDictionary_initial_benchmark_32MB_upper_limit_smarter_insertion,unified_string_dictionary_256MB_no_constraint]

CONFIG_BASE_NAME = "USSR_vs_MAIN"


def main() -> None:
    for pinned in PIN_VALUES:
        # build a benchmark where zipf_s is pinned to a single value
        benchmark = build_benchmark(s_values_list=[pinned])
        config: RunConfig = {
            "name": f"{CONFIG_BASE_NAME}_{PIN_VAR}{pinned}",
            "run_settings": RUN_SETTINGS,
            "system_settings": SYSTEM_SETTINGS,
            "systems": SYSTEMS,
            "benchmarks": benchmark,
        }
        run(config)


if __name__ == "__main__":
    main()
