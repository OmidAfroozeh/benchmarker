"""Driver script: benchmark **Zipf‑distributed** varchar data only.

Uniform can be approximated with ``zipf_s = 0`` — simply include that in the
`s_values` list.  All datasets are therefore generated with
``ColumnSpec.distribution = "zipf"`` while varying the *s* parameter.

Directory layout (unchanged except no more "uniform" folder):

    varchars_grp_size_zipf<s>/len_<L|Lhi‑lo>/varchars‑grp‑size‑<card>.db

Example paths
-------------
• Fixed length 48, *s = 0* (≈ uniform), 100 uniques →
    …/varchars_grp_size_zipf0/len_48/varchars-grp-size-100.db

• Variable length 8‑64, *s = 1.3*, 2000 uniques →
    …/varchars_grp_size_zipf1.3/len_8-64/varchars-grp-size-2000.db
"""

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

# ---------------------------------------------------------------------------
# Unified benchmark utilities
# ---------------------------------------------------------------------------
from synthetic_benchmark import ColumnSpec, generate_string_benchmark

# ---------------------------------------------------------------------------
# System definitions & experiment runner
# ---------------------------------------------------------------------------
from config.systems.duckdb import (
    UnifiedStringDictionary_lock_free_16mB,
    DUCK_DB_MAIN,
)
from src.models import DataSet, Benchmark, RunConfig, Query
from src.runner.experiment_runner import run

# ---------------------------------------------------------------------------
# get_data_path helper (fallback implementation outside repo)
# ---------------------------------------------------------------------------
try:
    from src.utils import get_data_path  # type: ignore
except Exception:  # pragma: no cover – standalone usage

    def get_data_path(rel: str) -> str:  # type: ignore
        return str(Path("/mnt/benchmarks") / rel)

# =============================================================================
# ░░ Parameter grids ░░
# =============================================================================
LengthSpec = Union[int, Tuple[int, int]]  # fixed or (min, max)

LENGTH_SPECS:    Sequence[LengthSpec] = [8, 16, (16,32)]
TOTAL_ROWS_LIST: Sequence[int]        = [100_000]
N_UNIQUE_LIST:   Sequence[int]        = [100]

# ─ Zipf *s* values ----------------------------------------------------------
S_VALUES: Sequence[float] = [0.0, 0.2]  # 0 ≈ uniform; add more as needed

# ─ Fixed generation knobs ---------------------------------------------------
CHUNK_ROWS:    int = 1_000_000
PARQUET_CODEC: str = "zstd"
SEED_BASE:     int = 999

# ----------------------------------------------------------------------------
# Custom query list (optional)
# ----------------------------------------------------------------------------
CUSTOM_QUERIES: List[Query] = [
    {
        "name": "single_column_groupby",
        "index": 0,
        "run_script": {"duckdb": "SELECT 1, str1 FROM varchars GROUP BY 1, str1"},
    }
]

# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

def len_spec_to_key(spec: LengthSpec) -> str:
    """Return identifier for filenames and dataset names."""
    if isinstance(spec, int):
        return str(spec)
    lo, hi = spec
    return f"{lo}-{hi}"


def build_db_path(len_spec: LengthSpec, n_unique: int, s_val: float) -> str:
    """Construct absolute DuckDB file path for parameters."""
    dist_dir = f"varchars_grp_size_zipf{s_val}"
    len_dir = f"len_{len_spec_to_key(len_spec)}"
    fname = f"varchars-grp-size-{n_unique}.db"
    rel = os.path.join(dist_dir, len_dir, fname)
    return get_data_path(rel)


def make_column_specs(spec: LengthSpec, n_unique: int, s_val: float) -> List[ColumnSpec]:
    """Return ColumnSpec list for one table variant (Zipf only)."""
    return [
        ColumnSpec("str1", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),
        ColumnSpec("str2", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),
        ColumnSpec("str3", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),

    ]

# ----------------------------------------------------------------------------
# Build datasets – Cartesian product over all grids
# ----------------------------------------------------------------------------

def assemble_datasets() -> List[DataSet]:
    datasets: List[DataSet] = []
    combo_iter = itertools.product(
        LENGTH_SPECS,
        TOTAL_ROWS_LIST,
        N_UNIQUE_LIST,
        S_VALUES,
    )

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
                f"ATTACH '{db_path}' (READ_ONLY); "
                f"USE '{db_path.stem}';"
            ),
        }

        name_key = len_spec_to_key(len_spec)
        datasets.append(
            {
                "name": f"len{name_key}_uni{uniques}_zipf{s_val}",
                "setup_script": setup_script,
                "config": {
                    "string_length": len_spec,
                    "n_unique": uniques,
                    "zipf_s": s_val,
                },
            }
        )
    return datasets

# ----------------------------------------------------------------------------
# Build benchmark descriptor
# ----------------------------------------------------------------------------

def build_benchmark() -> Benchmark:
    return {
        "name": "string_benchmark_zipf_grid",
        "datasets": assemble_datasets(),
        "queries": CUSTOM_QUERIES,
    }

# =============================================================================
# Runtime settings & system list
# =============================================================================
RUN_SETTINGS    = {"n_parallel": 1, "n_runs": 3}
SYSTEM_SETTINGS = [{"n_threads": 8}]
SYSTEMS         = [DUCK_DB_MAIN, UnifiedStringDictionary_lock_free_16mB]

# =============================================================================
# Main entry‑point
# =============================================================================

def main() -> None:
    benchmark = build_benchmark()

    config: RunConfig = {
        "name": "USSR_vs_baseline_zipf_grid",
        "run_settings": RUN_SETTINGS,
        "system_settings": SYSTEM_SETTINGS,
        "systems": SYSTEMS,
        "benchmarks": benchmark,
    }

    run(config)


if __name__ == "__main__":
    main()
