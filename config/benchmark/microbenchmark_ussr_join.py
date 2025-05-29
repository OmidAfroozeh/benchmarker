# =======================================================================
# File: config/benchmark/microbenchmark_strings_variable_length.py
# -----------------------------------------------------------------------
"""Refactored string‑length micro‑benchmark generator — *second revision*

All hard‑coded constants **and the data‑file locations** are now parameters
so you can set them exclusively from your driver script.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Callable

import duckdb  # noqa: F401  # Used by the helper but retained for clarity

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path

# ░░ String‑benchmark generator ░░ ###########################################
from .helper import ColumnSpec, generate_string_benchmark

logger = get_logger(__name__)

###############################################################################
# Default query list ##########################################################
###############################################################################

DEFAULT_MICRO_BENCHMARK_QUERY: List[Query] = [
    {
        "name": "double_column_groupby_1_constant",
        "index": 3,
        "run_script": {
            "duckdb": "select 1, str1 from varchars group by 1, str1",
        },
    },
]

###############################################################################
# Public factory ##############################################################
###############################################################################

def get_string_len_benchmark(
    string_lens: Sequence[int],
    *,
    total_rows: int = 100_000_000,
    n_unique: int = 1_000,
    chunk_rows: int = 5_000_000,
    seed_base: int = 42,
    benchmark_name: str = "string_micro_benchmark",
    micro_benchmark_query: List[Query] | None = None,
    # Path customisation ----------------------------------------------------------------
    db_root: str | os.PathLike | None = None,
    db_path_builder: Callable[[int], str] | None = None,
    # Schema customisation --------------------------------------------------------------
    column_specs_factory: Callable[[int, int], List[ColumnSpec]] | None = None,
) -> Benchmark:
    """Return a :class:`Benchmark` descriptor for the requested string lengths.

    **Everything** that used to be hard‑wired can now be passed in:

    * Generation knobs – *rows, uniques, chunk size, seed*
    * Query list — supply your own queries or reuse the default
    * Location of the DuckDB files via ``db_root`` **or** a full
      ``db_path_builder`` callable (the latter overrides the former)
    * Column schema through ``column_specs_factory``
    """

    queries = micro_benchmark_query or DEFAULT_MICRO_BENCHMARK_QUERY

    datasets = _generate_string_microbenchmark_data(
        string_lens=string_lens,
        total_rows=total_rows,
        n_unique=n_unique,
        chunk_rows=chunk_rows,
        seed_base=seed_base,
        db_root=db_root,
        db_path_builder=db_path_builder,
        column_specs_factory=column_specs_factory,
    )

    return {
        "name": benchmark_name,
        "datasets": datasets,
        "queries": queries,
    }

###############################################################################
# Helpers #####################################################################
###############################################################################

def _get_db_file_path(string_len: int, db_root: str | os.PathLike | None) -> str:
    """Default location mirroring the legacy layout."""

    relative = os.path.join(
        "varchars_variable_length", f"varchars-length-{string_len}.db"
    )
    return get_data_path(relative) if db_root is None else str(Path(db_root) / relative)


def _default_column_specs(string_len: int, n_unique: int) -> List[ColumnSpec]:
    """Three uniform dictionary‑encoded varchar columns (legacy schema)."""

    return [
        ColumnSpec("str1", n_unique, string_len, "uniform", use_dictionary=True),
        ColumnSpec("str2", n_unique, string_len, "uniform", use_dictionary=True),
        ColumnSpec("str3", n_unique, string_len, "uniform", use_dictionary=True),
    ]


def _generate_string_microbenchmark_data(
    *,
    string_lens: Sequence[int],
    total_rows: int,
    n_unique: int,
    chunk_rows: int,
    seed_base: int,
    db_root: str | os.PathLike | None,
    db_path_builder: Callable[[int], str] | None,
    column_specs_factory: Callable[[int, int], List[ColumnSpec]] | None,
) -> List[DataSet]:
    """Ensure all requested data sets exist; generate if necessary."""

    datasets: List[DataSet] = []
    column_specs_factory = column_specs_factory or _default_column_specs

    for idx, sl in enumerate(string_lens, start=1):
        logger.info("[%d/%d] Preparing data‑set for string length %d …", idx, len(string_lens), sl)

        # Fully custom path > db_root‑based path > legacy default sequence
        if db_path_builder is not None:
            duckdb_path = Path(db_path_builder(sl))
        else:
            duckdb_path = Path(_get_db_file_path(sl, db_root))

        duckdb_path.parent.mkdir(parents=True, exist_ok=True)

        if duckdb_path.exists():
            logger.info("  DuckDB %s already exists → skip generation", duckdb_path)
        else:
            logger.info("  Generating %s …", duckdb_path)

            generate_string_benchmark(
                duckdb_path=duckdb_path,
                total_rows=total_rows,
                column_specs=column_specs_factory(sl, n_unique),
                chunk_rows=chunk_rows,
                seed=seed_base + sl,  # distinct seed per variant
            )

        setup_script = {
            "duckdb": (
                f"ATTACH '{duckdb_path}' (READ_ONLY); "
                f"USE '{duckdb_path.stem}';"
            ),
        }

        datasets.append(
            {
                "name": f"varchars_len_{sl}",
                "setup_script": setup_script,
                "config": {
                    "string_length": sl,
                    "total_rows": total_rows,
                    "n_unique": n_unique,
                },
            }
        )

    return datasets

###############################################################################
# Quick test run ##############################################################
###############################################################################

if __name__ == "__main__":
    bm = get_string_len_benchmark([8, 16, 64], db_root="/tmp/duckdb_bench")
    print("Generated", len(bm["datasets"]), "datasets → first path:", bm["datasets"][0]["config"]["duckdb_path"])