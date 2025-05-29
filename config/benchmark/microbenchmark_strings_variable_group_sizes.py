"""
Generate and describe the variable-group-size varchar micro-benchmark.

The caller may specify *string_len* (fixed width of all varchar columns).
This value is propagated into:
    • the on-disk DuckDB file path
    • the dataset name
    • the benchmark metadata ("string_len" in dataset.config)
Default behaviour (caller omits *string_len*) is identical to the old
hard-coded 32-byte strings.
"""
import os
from typing import List

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path

# ---------------------------------------------------------------------------
# Helper utilities from standalone generator
# ---------------------------------------------------------------------------
from .helper import ColumnSpec, generate_string_benchmark  # adjust path if needed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Benchmark queries (unchanged)
# ---------------------------------------------------------------------------
MICRO_BENCHMARK_QUERY: List[Query] = [
    {
        "name": "single_column_groupby",
        "index": 0,
        "run_script": {
            "duckdb": "select str1 from varchars group by str1",
        },
    },
    {
        "name": "double_column_groupby",
        "index": 1,
        "run_script": {
            "duckdb": "select str1, str2 from varchars group by str1, str2",
        },
    },
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
DEFAULT_STRING_LEN: int = 32  # previous fixed width

TOTAL_ROWS: int = 50_000_000         # rows per dataset
PARQUET_CHUNK_ROWS: int = 5_000_000   # Parquet chunk size
SEED: int = 42                        # RNG seed for deterministic builds


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def get_string_len_benchmark(
    group_sizes: List[int],
    *,
    string_len: int = DEFAULT_STRING_LEN,
) -> Benchmark:
    """
    Build a complete benchmark descriptor for the given *group_sizes*
    (cardinalities) and fixed-width *string_len*.
    """
    datasets: List[DataSet] = _generate_and_collect_datasets(group_sizes, string_len)

    return {
        "name": f"string_micro_benchmark_len{string_len}",
        "datasets": datasets,
        "queries": MICRO_BENCHMARK_QUERY,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _benchmark_db_path(cardinality: int, string_len: int) -> str:
    """
    Absolute path to the DuckDB database file for the given (cardinality, string_len)
    combination.

    Layout example:
        …/varchars_grp_size_uniform/len_48/varchars-grp-size-100.db
    """
    rel = os.path.join(
        "varchars_grp_size_uniform",
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
    """
    Create (if necessary) a DuckDB database holding three independent string columns.

    Each column shares the same *cardinality* but draws values from an independent
    pool so that str1, str2 and str3 are uncorrelated.  The fixed length is *string_len*.
    """
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
            distribution="uniform",
            use_dictionary=True,
        ),
        ColumnSpec(
            name="str2",
            n_unique=cardinality,
            str_len=string_len,
            distribution="uniform",
            use_dictionary=True,
        ),
        ColumnSpec(
            name="str3",
            n_unique=cardinality,
            str_len=string_len,
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
