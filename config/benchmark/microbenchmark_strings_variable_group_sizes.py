import os
from typing import List

import duckdb

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path

# ---------------------------------------------------------------------------
# Helper utilities from standalone generator
# ---------------------------------------------------------------------------
from .helper import ColumnSpec, generate_string_benchmark  # adjust the import path if needed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Benchmark queries (unchanged)
# ---------------------------------------------------------------------------
MICRO_BENCHMARK_QUERY: List[Query] = [
    {
        "name": "single_column_groupby",
        "index": 0,
        "run_script": {
            "duckdb": "SELECT str1 FROM varchars GROUP BY str1",
        },
    },
    {
        "name": "double_column_groupby",
        "index": 1,
        "run_script": {
            "duckdb": "SELECT str1, str2 FROM varchars GROUP BY str1, str2",
        },
    },
]

# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_string_len_benchmark(string_lens: List[int]) -> Benchmark:
    """Return a complete benchmark descriptor for the given cardinalities."""

    datasets: List[DataSet] = _generate_and_collect_datasets(string_lens)

    return {
        "name": "string_micro_benchmark",
        "datasets": datasets,
        "queries": MICRO_BENCHMARK_QUERY,
    }

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _benchmark_db_path(cardinality: int) -> str:
    rel = os.path.join("varchars_grp_size_uniform", f"varchars-grp-size-{cardinality}.db")
    return get_data_path(rel)


TOTAL_ROWS: int = 10_000_000        # Rows per data‑set
PARQUET_CHUNK_ROWS: int = 1_000_000  # Parquet chunk size
STRING_LEN: int = 32                 # Fixed‑width strings for both columns
SEED: int = 42                       # RNG seed for deterministic builds


def _generate_and_collect_datasets(cardinalities: List[int]) -> List[DataSet]:
    datasets: List[DataSet] = []
    for card in cardinalities:
        db_path = _benchmark_db_path(card)
        _ensure_dataset(card, db_path)

        db_name = os.path.splitext(os.path.basename(db_path))[0]
        setup_script = {
            "duckdb": f"ATTACH '{db_path}' (READ_ONLY); USE '{db_name}';",
        }

        datasets.append(
            {
                "name": f"tpcds-{card}",
                "setup_script": setup_script,
                "config": {"group_size": card},
            }
        )
    return datasets


def _ensure_dataset(cardinality: int, db_path: str) -> None:
    """Create a DuckDB database holding two **different** string columns.

    *Both* columns share the same cardinality but draw their values from
    independent pools, ensuring that `str1` and `str2` do **not** contain the
    same strings row‑wise.
    """

    if os.path.exists(db_path):
        logger.info("File %s already exists → skip generation", db_path)
        return

    logger.info("Generating data for cardinality %d → %s", cardinality, db_path)

    col_specs = [
        ColumnSpec(
            name="str1",
            n_unique=cardinality,
            str_len=STRING_LEN,
            distribution="uniform",
            use_dictionary=True,
        ),
        ColumnSpec(
            name="str2",
            n_unique=cardinality,
            str_len=STRING_LEN,
            distribution="uniform",
            use_dictionary=True,
        ),
        ColumnSpec(
            name="str3",
            n_unique=cardinality,
            str_len=STRING_LEN,
            distribution="uniform",
            use_dictionary=True,
        ),
    ]

    generate_string_benchmark(
        duckdb_path=db_path,
        total_rows=TOTAL_ROWS,
        column_specs=col_specs,
        chunk_rows=PARQUET_CHUNK_ROWS,
        parquet_codec="zstd",
        seed=SEED,
        cleanup=True,
        global_unique=False,
    )

    logger.info("Finished → %s", db_path)
