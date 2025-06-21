from __future__ import annotations

"""String benchmark runner with *two* pinned parameters.

This variant allows you to pin both the Zipf skew (``zipf_s``)
*and* the string‑length specification so that those parameters do
not take part in the Cartesian product when generating datasets.
"""

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

# -- Global grids (used when parameter is *not* pinned) -----------------------
LENGTH_SPECS_FULL: Sequence[LengthSpec] = [16, 32]
DEFAULT_S_VALUES: Sequence[float] = [0.0]

# -- Pinned parameters --------------------------------------------------------
#   Define *lists* so we can easily iterate over the Cartesian product
PIN_ZIPF_VALUES: Sequence[float] = [1.0]      # `zipf_s` will be fixed to these values
PIN_LENGTH_SPECS: Sequence[LengthSpec] = [16, 32]  # string lengths to pin

# -- Remaining generation knobs ----------------------------------------------
TOTAL_ROWS_LIST: Sequence[int] = [20_000_000]
N_UNIQUE_LIST: Sequence[int] = [1000, 10_000, 50_000, 100_000, 350_000, 500_000, 1_000_000]

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
    dist_dir = (
        f"varchars_grp_size_zipf={s_val}_nrows={TOTAL_ROWS_LIST[0] / 1_000_000}M_"
        f"uniques={n_unique}_len={len_spec_to_key(len_spec)}"
    )
    fname = f"varchars-grp-size-{n_unique}.db"
    rel = os.path.join(dist_dir, fname)
    return get_data_path(rel)


def make_column_specs(spec: LengthSpec, n_unique: int, s_val: float) -> List[ColumnSpec]:
    return [
        ColumnSpec("str1", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),
        ColumnSpec("str2", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),
    ]


def assemble_datasets(
    *,
    length_specs: Sequence[LengthSpec],
    total_rows_list: Sequence[int] = TOTAL_ROWS_LIST,
    n_unique_list: Sequence[int] = N_UNIQUE_LIST,
    s_values: Sequence[float] = DEFAULT_S_VALUES,
) -> List[DataSet]:
    """Assemble datasets for the Cartesian product of the provided parameter lists.

    Any parameter that you want to *pin* should simply be passed as a singleton
    list (e.g. ``length_specs=[16]``).
    """
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
                "config": {"string_length": len_spec, "n_unique": uniques},
            }
        )
    return datasets


# =============================================================================
# ░░ Benchmark builder ░░
# =============================================================================

def build_benchmark(*, length_specs: Sequence[LengthSpec], s_values_list: Sequence[float]) -> Benchmark:
    """Build a Benchmark object with the provided (possibly pinned) parameter lists."""
    return {
        "name": "string_benchmark",
        "datasets": assemble_datasets(length_specs=length_specs, s_values=s_values_list),
        "queries": CUSTOM_QUERIES,
    }


# =============================================================================
# ░░ Main entry‑point ░░
# =============================================================================
RUN_SETTINGS = {"n_parallel": 1, "n_runs": 6}
SYSTEM_SETTINGS = [{"n_threads": 8}]
SYSTEMS = [
    DUCK_DB_MAIN,
    UnifiedStringDictionary_1GB_full_insertion,
    UnifiedStringDictionary_initial_benchmark_32MB_upper_limit_smarter_insertion,
]
CONFIG_BASE_NAME = "USSR_vs_MAIN"


def main() -> None:
    """Launch experiments for *all* combinations of pinned values."""
    for s_val, len_spec in itertools.product(PIN_ZIPF_VALUES, PIN_LENGTH_SPECS):
        benchmark = build_benchmark(length_specs=[len_spec], s_values_list=[s_val])

        config_name = (
            f"{CONFIG_BASE_NAME}_zipf{s_val}_len{len_spec_to_key(len_spec)}"
        )
        config: RunConfig = {
            "name": config_name,
            "run_settings": RUN_SETTINGS,
            "system_settings": SYSTEM_SETTINGS,
            "systems": SYSTEMS,
            "benchmarks": benchmark,
        }
        run(config)


if __name__ == "__main__":
    main()
