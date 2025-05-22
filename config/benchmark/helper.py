import os
import math
import logging
import shutil
import string as _string
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union, Literal, Optional, Dict, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

logger = logging.getLogger("string_benchmark")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
logger.addHandler(handler)

###############################################################################
# Column specification ########################################################
###############################################################################

DistributionLiteral = Literal["uniform", "zipf"]

@dataclass
class ColumnSpec:
    """Describe one string column for the benchmark data‑set.

    *Guarantees* **exactly `n_unique` distinct strings per column** across the
    *whole table* (not per chunk). Duplication between columns is allowed by
    default; pass ``global_unique=True`` to the generator to forbid that.
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
# Generator ###################################################################
###############################################################################

def generate_string_benchmark(
    duckdb_path: Union[str, Path],
    total_rows: int,
    column_specs: List[ColumnSpec],
    *,
    chunk_rows: int = 5_000_000,
    parquet_codec: str = "zstd",
    alphabet: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    cleanup: bool = True,
    global_unique: bool = False,
) -> Path:
    """Create Parquet chunk set + DuckDB database for string benchmarks.

    Guarantees each column has exactly ``n_unique`` distinct values across the
    *whole* table, regardless of chunk size. The check that raised for your
    2 M‑row chunk is now gone because uniqueness is enforced table‑wide.
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
        alphabet = np.array(list(_string.ascii_lowercase), dtype="U1")

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
    con.execute("PRAGMA force_compression='dictionary'")
    col_list = ", ".join(spec.name for spec in column_specs)
    con.execute(f"CREATE TABLE varchars AS SELECT {col_list} FROM read_parquet('{parquet_dir}/*.parquet')")
    con.execute("CHECKPOINT")
    con.close()
    logger.info("DuckDB ready → %s", duckdb_path)

    if cleanup:
        shutil.rmtree(parquet_dir, ignore_errors=True)
        logger.info("Removed %s", parquet_dir)

    return duckdb_path

if __name__ == "__main__":

    cols = [
        # Fully dictionary-friendly, low cardinality, fixed length
        ColumnSpec(
            name="dict_low_card",
            n_unique=1_000,
            str_len=64,
            distribution="uniform",
            use_dictionary=True
        ),

        # High-cardinality column with Zipfian skew and variable length
        ColumnSpec(
            name="zipf_hi_card",
            n_unique=5_000_000,
            str_len=64,       # strings range from 8–64 chars
            distribution="uniform",
            use_dictionary=True
        ),

        # Medium cardinality, uniform distribution, *no* dictionary hint
        ColumnSpec(
            name="uniform_midcard_no_dict",
            n_unique=50_000,
            str_len=32,
            distribution="uniform",
            use_dictionary=False
        ),
    ]

    # -------------------------------------------------------------------
    # 2️⃣  Generate the data set
    # -------------------------------------------------------------------
    db_path = Path("bench_100smM.db")
    generate_string_benchmark(
        duckdb_path=db_path,
        total_rows=100_000_000,   # 10 million rows
        column_specs=cols,
        chunk_rows=2_000_000,    # optional: smaller Parquet chunks
        parquet_codec="zstd",    # default; you could use "snappy" etc.
        seed=42                  # deterministic randomness for testing
    )

    print(f"Done → DuckDB at {db_path} (Parquet in {db_path.with_suffix('').with_name(db_path.stem + '_parquet')})")