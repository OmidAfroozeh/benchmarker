"""
Zipf(1.2) variable-group-size micro-benchmark.

Callers supply *string_len*; default is the old hard-coded width (64).
The string length is embedded in:
    • the DuckDB file path     …/varchars_grp_size_zipf1.2/len_<N>/…
    • the dataset name         tpcds-<card>-len<N>
    • the dataset.config entry {"group_size": …, "string_len": N}
"""

import os
from typing import List

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path

# ---------------------------------------------------------------------------
# Helper utilities from standalone generator
# ---------------------------------------------------------------------------
from .helper import ColumnSpec, generate_string_benchmark  # adjust if needed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Benchmark queries (unchanged)
# ---------------------------------------------------------------------------
MICRO_BENCHMARK_QUERY: List[Query] = [
    {
        "name": "double_column_groupby_1_constant",
        "index": 3,
        "run_script": {
            "duckdb": "select 1, str1 from varchars group by 1, str1",
        },
    },
]

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_STRING_LEN: int = 64

TOTAL_ROWS: int = 50_000_000       # rows per dataset
PARQUET_CHUNK_ROWS: int = 2_000_000
SEED: int = 42


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def get_string_len_benchmark(
    group_sizes: List[int],
    *,
    string_len: int = DEFAULT_STRING_LEN,
) -> Benchmark:
    datasets = _generate_and_collect_datasets(group_sizes, string_len)

    return {
        "name": f"string_micro_benchmark_zipf_len{string_len}",
        "datasets": datasets,
        "queries": MICRO_BENCHMARK_QUERY,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _benchmark_db_path(cardinality: int, string_len: int) -> str:
    """Return absolute DB path for (cardinality, string_len)."""
    rel = os.path.join(
        "varchars_grp_size_zipf0.8",
        f"len_{string_len}",
        f"varchars-grp-size-{cardinality}.db",
    )
    return get_data_path(rel)


def _generate_and_collect_datasets(
    cardinalities: List[int],
    string_len: int,
) -> List[DataSet]:
    datasets: List[DataSet] = []
    for card in cardinalities:
        db_path = _benchmark_db_path(card, string_len)
        _ensure_dataset(card, string_len, db_path)

        db_name = os.path.splitext(os.path.basename(db_path))[0]
        setup_script = {"duckdb": f"ATTACH '{db_path}' (READ_ONLY); USE '{db_name}';"}

        datasets.append(
            {
                "name": f"tpcds-{card}-len{string_len}",
                "setup_script": setup_script,
                "config": {"group_size": card, "string_len": string_len},
            }
        )
    return datasets


def _ensure_dataset(
    cardinality: int,
    string_len: int,
    db_path: str,
) -> None:
    """Generate a DuckDB file with three independent Zipf-distributed columns."""

    if os.path.exists(db_path):
        logger.info("File %s already exists → skip generation", db_path)
        return

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    logger.info(
        "Generating data for cardinality %d (len=%d) → %s",
        cardinality,
        string_len,
        db_path,
    )

    col_specs = [
        ColumnSpec(
            name="str1",
            n_unique=cardinality,
            str_len=string_len,
            distribution="zipf",
            zipf_s=0.9,
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
