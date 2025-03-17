from __future__ import annotations

import logging
import math
import os
import shutil
import string as _string
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
)

import duckdb  # noqa: F401 – used by generated queries / setup scripts
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query


logger = get_logger(__name__)

###############################################################################
# Column specification ########################################################
###############################################################################

DistributionLiteral = Literal["uniform", "zipf"]


@dataclass
class ColumnSpec:
    """Describe one string column for the benchmark data‑set.

    *Guarantees* **exactly ``n_unique`` distinct strings per column** across
    the *whole table* (not per chunk). Duplication between columns is allowed
    unless ``global_unique=True`` is used in :func:`generate_string_benchmark`.
    """

    name: str
    n_unique: int
    str_len: Union[int, Tuple[int, int]]
    distribution: DistributionLiteral = "uniform"
    zipf_s: float = 1.3
    use_dictionary: bool = True

    # internal caches ----------------------------------------------------
    _pool: np.ndarray = field(init=False, repr=False, default=None)
    _indices: Dict[int, np.ndarray] = field(init=False, repr=False, default_factory=dict)

    # ------------------------------------------------------------------
    # Unique‑string pool (guaranteed cardinality)
    # ------------------------------------------------------------------
    def _generate_unique_pool(self, alphabet: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        generated: Set[str] = set()
        pool: List[str] = []

        def _draw_batch(size: int) -> List[str]:
            if isinstance(self.str_len, int):
                chars = rng.choice(alphabet, size=(size, self.str_len))
                return ["".join(row) for row in chars]
            lo, hi = self.str_len  # type: ignore[misc]
            lens = rng.integers(lo, hi + 1, size=size)
            out: List[str] = []
            for L in np.unique(lens):
                cnt = np.sum(lens == L)
                chars = rng.choice(alphabet, size=(cnt, L))
                out.extend("".join(row) for row in chars)
            return out

        batch_size = max(50_000, self.n_unique // 20)
        while len(pool) < self.n_unique:
            need = self.n_unique - len(pool)
            size = int(need * 1.2) if need < batch_size else batch_size
            for s in _draw_batch(size):
                if s not in generated:
                    generated.add(s)
                    pool.append(s)
                    if len(pool) == self.n_unique:
                        break
        return np.asarray(pool, dtype="O")

    def ensure_pool(self, alphabet: np.ndarray):
        if self._pool is None:
            self._pool = self._generate_unique_pool(alphabet)

    # ------------------------------------------------------------------
    # Pre‑compute *table‑wide* index arrays (one per total_rows value)
    # ------------------------------------------------------------------
    def _build_indices_uniform(self, total_rows: int) -> np.ndarray:
        rows_per_val, remainder = divmod(total_rows, self.n_unique)
        idx = np.repeat(np.arange(self.n_unique), rows_per_val)
        if remainder:
            idx = np.concatenate([idx, np.arange(remainder)])
        np.random.shuffle(idx)
        return idx

    def _build_indices_zipf(self, total_rows: int) -> np.ndarray:
        ranks = np.arange(1, self.n_unique + 1)
        probs = 1.0 / (ranks ** self.zipf_s)
        probs /= probs.sum()
        # mandatory one‑each coverage
        idx = np.arange(self.n_unique)
        remaining = total_rows - self.n_unique
        if remaining > 0:
            extra = np.random.choice(self.n_unique, size=remaining, p=probs)
            idx = np.concatenate([idx, extra])
        np.random.shuffle(idx)
        return idx

    def ensure_indices(self, total_rows: int):
        if total_rows in self._indices:
            return
        if total_rows < self.n_unique:
            raise ValueError(
                f"total_rows ({total_rows:,}) < n_unique ({self.n_unique:,}) for column '{self.name}'"
            )
        if self.distribution == "uniform":
            self._indices[total_rows] = self._build_indices_uniform(total_rows)
        elif self.distribution == "zipf":
            self._indices[total_rows] = self._build_indices_zipf(total_rows)
        else:
            raise ValueError("Unknown distribution")

    def slice_chunk(self, total_rows: int, start: int, end: int) -> np.ndarray:
        self.ensure_indices(total_rows)
        idx_arr = self._indices[total_rows][start:end]
        return self._pool[idx_arr]


###############################################################################
# Low‑level generator #########################################################
###############################################################################

def generate_string_benchmark(
    *,
    duckdb_path: Union[str, Path],
    total_rows: int,
    column_specs: List[ColumnSpec],
    chunk_rows: int = 5_000_000,
    parquet_codec: str = "zstd",
    alphabet: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    cleanup: bool = True,
    global_unique: bool = False,
) -> Path:
    """Create Parquet chunk set **plus** a DuckDB file holding one table
    named ``varchars``.

    Parameters
    ----------
    duckdb_path: str | Path
        Target DuckDB database file. Generation is skipped if the file exists.
    total_rows: int
        Number of rows in the final table.
    column_specs: List[ColumnSpec]
        Column definitions (see :class:`ColumnSpec`).
    chunk_rows: int, default 5_000_000
        Row count per Parquet chunk file.
    parquet_codec: str, default "zstd"
        Compression codec for Parquet files.
    alphabet: np.ndarray[str], optional
        Alphabet from which random characters are drawn. Defaults to
        ``ascii_lowercase``.
    seed: int, optional
        Seed the RNG for reproducible datasets.
    cleanup: bool, default True
        Remove the temporary Parquet directory after loading data into DuckDB.
    global_unique: bool, default False
        When *True*, the sets of unique strings of **all** columns must be
        disjoint.
    """

    # sanity -------------------------------------------------------------
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if chunk_rows <= 0 or chunk_rows > total_rows:
        raise ValueError("chunk_rows must be in (0, total_rows]")
    if not column_specs:
        raise ValueError("Need at least one ColumnSpec")

    duckdb_path = Path(duckdb_path)
    if duckdb_path.exists():
        logger.info("DuckDB %s already exists → skip", duckdb_path)
        return duckdb_path

    parquet_dir = duckdb_path.with_suffix("").with_name(duckdb_path.stem + "_parquet")
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # RNG & alphabet -----------------------------------------------------
    if seed is not None:
        np.random.seed(seed)
        logger.info("Using seed = %d", seed)

    if alphabet is None:
        # alphabet = np.array(list(_string.ascii_lowercase), dtype="U1")
        alphabet = np.array(list(_string.ascii_letters + _string.digits + _string.punctuation), dtype="U1")

    # pools --------------------------------------------------------------
    used: Set[str] = set()
    for spec in column_specs:
        if global_unique:
            while True:
                spec.ensure_pool(alphabet)
                if not used.intersection(spec._pool):
                    break
                spec._pool = None  # regenerate
            used.update(spec._pool)
        else:
            spec.ensure_pool(alphabet)
        spec.ensure_indices(total_rows)
        logger.info("Column %-25s → %8d uniques", spec.name, spec.n_unique)

    # write Parquet ------------------------------------------------------
    n_chunks = math.ceil(total_rows / chunk_rows)
    logger.info("Writing %d chunk(s) to %s", n_chunks, parquet_dir)
    for c in range(n_chunks):
        start = c * chunk_rows
        end = min(start + chunk_rows, total_rows)
        df = pd.DataFrame({
            spec.name: spec.slice_chunk(total_rows, start, end) for spec in column_specs
        }, copy=False)
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            parquet_dir / f"chunk_{c:05d}.parquet",
            compression=parquet_codec,
            use_dictionary={spec.name: spec.use_dictionary for spec in column_specs},
            write_statistics=False,
            data_page_size=1 << 20,
        )
        logger.info("  chunk %d/%d (%d rows) done", c + 1, n_chunks, end - start)

    logger.info("Parquet ready → %s", parquet_dir)

    # load into DuckDB ----------------------------------------------------
    logger.info("Creating DuckDB → %s", duckdb_path)
    con = duckdb.connect(duckdb_path)
    # con.execute("PRAGMA force_compression='dictionary'")
    col_list = ", ".join(spec.name for spec in column_specs)
    con.execute(f"CREATE TABLE varchars AS SELECT {col_list} FROM read_parquet('{parquet_dir}/*.parquet')")
    con.execute("CHECKPOINT")
    con.close()
    logger.info("DuckDB ready → %s", duckdb_path)

    if cleanup:
        shutil.rmtree(parquet_dir, ignore_errors=True)
        logger.info("Removed %s", parquet_dir)

    return duckdb_path


###############################################################################
# High‑level micro‑benchmark factory #########################################
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


def _get_db_file_path(string_len: int, db_root: str | os.PathLike | None) -> str:
    """Default on‑disk location mirroring the legacy repo layout."""

    relative = os.path.join(
        "varchars_variable_length", f"varchars-length-{string_len}.db"
    )
    if db_root is None:
        # lazily import util only when needed
        try:
            from src.utils import get_data_path  # type: ignore

            return get_data_path(relative)
        except Exception:
            return str(Path.cwd() / relative)
    return str(Path(db_root) / relative)


def _default_column_specs(string_len: int, n_unique: int) -> List[ColumnSpec]:
    """Legacy schema: three uniform dictionary‑encoded columns."""

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
    """Ensure all requested data‑sets exist; generate them if necessary."""

    datasets: List[DataSet] = []
    column_specs_factory = column_specs_factory or _default_column_specs

    for idx, sl in enumerate(string_lens, start=1):
        logger.info("[%d/%d] Preparing data‑set for string length %d …", idx, len(string_lens), sl)

        # Fully custom path > db_root‑based path > fallback default path
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
                    "chunk_rows": chunk_rows,
                    "duckdb_path": str(duckdb_path),
                },
            }
        )

    return datasets


def generate_synthetic_string_benchmarks(
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
    """Return a :class:`Benchmark` descriptor for the given *string_lens*.

    Everything that used to be hard‑coded can now be supplied via arguments:

    * Generation knobs – *rows, uniques, chunk size, seed*
    * Query list – pass your own queries or reuse the defaults
    * Location of the DuckDB files via ``db_root`` **or** a custom
      ``db_path_builder`` (overrides the former)
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


__all__ = [
    "ColumnSpec",
    "generate_string_benchmark",
    "generate_synthetic_string_benchmarks",
]
