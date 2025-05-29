import os
from pathlib import Path
from typing import List

import duckdb

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path

# ░░ String‑benchmark generator ░░ ###########################################
from .helper import ColumnSpec, generate_string_benchmark

logger = get_logger(__name__)

###############################################################################
# Queries used by the micro‑benchmark ########################################
###############################################################################

MICRO_BENCHMARK_QUERY: List[Query] = [
    # {
    #     "name": "single_column_groupby",
    #     "index": 0,
    #     "run_script": {
    #         "duckdb": "select str1 from varchars group by str1",
    #     },
    # },
    # {
    #     "name": "double_column_groupby",
    #     "index": 1,
    #     "run_script": {
    #         "duckdb": "select str1, str2 from varchars group by str1, str2",
    #     },
    # },
    # {
    #     "name": "triple_column_groupby",
    #     "index": 2,
    #     "run_script": {
    #         "duckdb": "select str1, str2, str3 from varchars group by str1, str2, str3",
    #     },
    # },
    {
        "name": "double_column_groupby_1_constant",
        "index": 3,
        "run_script": {
            "duckdb": "select 1, str1 from varchars group by 1, str1",
        },
    },
]

###############################################################################
# Public API ##################################################################
###############################################################################

def get_string_len_benchmark(string_lens: List[int]) -> Benchmark:
    """Return a Benchmark descriptor for the requested string‑length variants."""

    datasets: List[DataSet] = _generate_string_microbenchmark_data(string_lens)

    return {
        "name": "string_micro_benchmark",
        "datasets": datasets,
        "queries": MICRO_BENCHMARK_QUERY,
    }

###############################################################################
# Helpers #####################################################################
###############################################################################

def _get_db_file_path(string_len: int) -> str:
    """Consistent location for the generated DuckDB databases."""
    relative = os.path.join("varchars_variable_length", f"varchars-length-{string_len}.db")
    return get_data_path(relative)


def _generate_string_microbenchmark_data(string_lens: List[int]) -> List[DataSet]:
    """Ensure all requested data‑sets exist, generating them if necessary."""

    TOTAL_ROWS = 100_000_000  # keep the original row count
    N_UNIQUE   = 1_000       # matches the base implementation (10 M / 10 k)
    CHUNK_ROWS = 5_000_000   # reasonable 2 M‑row chunks

    datasets: List[DataSet] = []

    for idx, sl in enumerate(string_lens, start=1):
        logger.info("[%d/%d] Preparing data‑set for string length %d …", idx, len(string_lens), sl)
        duckdb_path = Path(_get_db_file_path(sl))
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)

        if duckdb_path.exists():
            logger.info("  DuckDB %s already exists → skip generation", duckdb_path)
        else:
            logger.info("  Generating %s …", duckdb_path)

            # Build two identical columns (str1, str2) to replicate legacy schema.
            columns = [
                ColumnSpec("str1", N_UNIQUE, sl, "uniform", use_dictionary=True),
                ColumnSpec("str2", N_UNIQUE, sl, "uniform", use_dictionary=True),
                ColumnSpec("str3", N_UNIQUE, sl, "uniform", use_dictionary=True),
            ]

            generate_string_benchmark(
                duckdb_path=duckdb_path,
                total_rows=TOTAL_ROWS,
                column_specs=columns,
                chunk_rows=CHUNK_ROWS,
                seed=42 + sl,  # change seed per length for variety
            )

        # Compose DataSet entry expected by benchmark harness
        setup_script = {
            "duckdb": f"ATTACH '{duckdb_path}' (READ_ONLY); USE '{duckdb_path.stem}';",
        }

        datasets.append({
            "name": f"varchars_len_{sl}",
            "setup_script": setup_script,
            "config": {"string_length": sl},
        })

    return datasets

###############################################################################
# Module test run #############################################################
###############################################################################

if __name__ == "__main__":
    # Quick sanity check when run directly
    bm = get_string_len_benchmark([8, 16, 64])
    print("Generated benchmark descriptor with", len(bm["datasets"]), "datasets.")
