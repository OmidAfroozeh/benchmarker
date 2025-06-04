"""string_benchmark_multi_tables_driver_no_cli.py – Generator *plus* two‑table benchmark driver
with non‑sequential, overlapped IDs and **configurable constant join cardinality**
================================================================================
Standalone **no‑CLI** version of the original driver: simply tweak the constants
below (especially `EXPECTED_JOIN_ROWS_DEFAULT`, `ROOT_OVERRIDE`, and
`RUN_BENCHMARK`) and run the file.  All command‑line argument parsing has been
removed.

Key differences vs. the original
--------------------------------
1. **No `argparse` / CLI switches** – everything is configured in‑file.
2. Execution is controlled by a few top‑level constants (see "Runtime control"
   section).  Changing a constant automatically propagates everywhere.
3. When run as a script (`python string_benchmark_multi_tables_driver_no_cli.py`)
   it will (optionally) generate the data **and** launch the benchmark in a
   single go.  Flip `RUN_BENCHMARK = False` to skip the experiment runner and
   only build the datasets.
"""

from __future__ import annotations

##############################################################################
# ░░ Imports & fallback stubs ░░                                              #
##############################################################################

import logging
import math
import os
import shutil
import string as _string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
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

import duckdb  # runtime dependency for the generated DB files
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Optional repo‑internal helpers ──────────────────────────────────────────
try:
    from src.logger import get_logger  # type: ignore
except Exception:  # pragma: no cover – standalone mode

    def get_logger(name: str):  # type: ignore
        _logger = logging.getLogger(name)
        if not _logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
        return _logger

try:
    from src.models import Benchmark, DataSet, Query, RunConfig  # type: ignore
except Exception:  # pragma: no cover – standalone mode

    class Query(TypedDict, total=False):
        name: str
        index: int
        run_script: Dict[str, str]

    class DataSet(TypedDict, total=False):
        name: str
        setup_script: Dict[str, str]
        config: Dict[str, object]

    class Benchmark(TypedDict, total=False):
        name: str
        datasets: List[DataSet]
        queries: List[Query]

    class RunConfig(TypedDict, total=False):
        name: str
        run_settings: Dict[str, object]
        system_settings: List[Dict[str, object]]
        systems: List[object]
        benchmarks: Benchmark

logger = get_logger(__name__)

##############################################################################
# ░░ Part 1 – Multi‑table string generator ░░                                #
##############################################################################

DistributionLiteral = Literal["uniform", "zipf"]
ID_COL = "id"  # primary‑key column shared by both tables


@dataclass
class ColumnSpec:
    """Describe one varchar column – guarantees exactly *n_unique* values."""

    name: str
    n_unique: int
    str_len: Union[int, Tuple[int, int]]
    distribution: DistributionLiteral = "uniform"
    zipf_s: float = 1.3
    use_dictionary: bool = True

    # internal caches (not part of the public API) ---------------------------
    _pool: np.ndarray = field(init=False, repr=False, default=None)
    _indices: Dict[int, np.ndarray] = field(init=False, repr=False, default_factory=dict)

    # – helper: generate the unique string pool –
    def _generate_unique_pool(self, alphabet: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        pool: List[str] = []
        generated: Set[str] = set()

        def _draw_batch(size: int) -> List[str]:
            if isinstance(self.str_len, int):
                chars = rng.choice(alphabet, size=(size, self.str_len))
                return ["".join(row) for row in chars]
            lo, hi = self.str_len  # type: ignore[misc]
            lens = rng.integers(lo, hi + 1, size=size)
            out: List[str] = []
            for L in np.unique(lens):
                sub = rng.choice(alphabet, size=(np.sum(lens == L), L))
                out.extend("".join(r) for r in sub)
            return out

        batch_size = max(50_000, self.n_unique // 20)
        while len(pool) < self.n_unique:
            needed = self.n_unique - len(pool)
            size = int(needed * 1.2) if needed < batch_size else batch_size
            for s in _draw_batch(size):
                if s not in generated:
                    generated.add(s)
                    pool.append(s)
                    if len(pool) == self.n_unique:
                        break
        return np.asarray(pool, dtype="O")

    # public helpers ---------------------------------------------------------
    def ensure_pool(self, alphabet: np.ndarray):
        if self._pool is None:
            self._pool = self._generate_unique_pool(alphabet)

    def _build_indices_uniform(self, total_rows: int) -> np.ndarray:
        rows_per_val, rem = divmod(total_rows, self.n_unique)
        idx = np.repeat(np.arange(self.n_unique), rows_per_val)
        if rem:
            idx = np.concatenate([idx, np.arange(rem)])
        np.random.shuffle(idx)
        return idx

    def _build_indices_zipf(self, total_rows: int) -> np.ndarray:
        ranks = np.arange(1, self.n_unique + 1)
        probs = 1.0 / (ranks ** self.zipf_s)
        probs /= probs.sum()
        idx = np.arange(self.n_unique)  # mandatory one‑each coverage
        remaining = total_rows - self.n_unique
        if remaining:
            extra = np.random.choice(self.n_unique, size=remaining, p=probs)
            idx = np.concatenate([idx, extra])
        np.random.shuffle(idx)
        return idx

    def ensure_indices(self, total_rows: int):
        if total_rows in self._indices:
            return
        if total_rows < self.n_unique:
            raise ValueError("total_rows < n_unique for column %s" % self.name)
        if self.distribution == "uniform":
            self._indices[total_rows] = self._build_indices_uniform(total_rows)
        elif self.distribution == "zipf":
            self._indices[total_rows] = self._build_indices_zipf(total_rows)
        else:
            raise ValueError("unknown distribution")

    def slice_chunk(self, total_rows: int, start: int, end: int) -> np.ndarray:
        self.ensure_indices(total_rows)
        return self._pool[self._indices[total_rows][start:end]]


# ── Top‑level generator ─────────────────────────────────────────────────----


def generate_string_benchmark(
    *,
    duckdb_path: Union[str, Path],
    total_rows: int,
    column_specs: List[ColumnSpec],
    table_name: str = "varchars",
    id_values: Optional[np.ndarray] = None,
    if_exists: Literal["skip", "replace", "error"] = "skip",
    chunk_rows: int = 5_000_000,
    parquet_codec: str = "zstd",
    alphabet: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    cleanup: bool = True,
    global_unique: bool = False,
) -> Path:
    """Create (or append) a Parquet‑backed varchar table with a non‑sequential `id`."""

    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if not column_specs:
        raise ValueError("Need at least one ColumnSpec")
    if id_values is not None and len(id_values) != total_rows:
        raise ValueError("id_values length must equal total_rows")

    duckdb_path = Path(duckdb_path)
    parquet_dir = duckdb_path.with_suffix("").with_name(f"{table_name}_parquet")

    # RNG & alphabet -------------------------------------------------------
    if seed is not None:
        np.random.seed(seed)
        logger.info("[seed=%d]", seed)
    alphabet = alphabet or np.array(list(_string.ascii_lowercase), dtype="U1")

    # pools / indices ------------------------------------------------------
    used: Set[str] = set()
    for spec in column_specs:
        if global_unique:
            while True:
                spec.ensure_pool(alphabet)
                if not used.intersection(spec._pool):
                    break
                spec._pool = None
            used.update(spec._pool)
        else:
            spec.ensure_pool(alphabet)
        spec.ensure_indices(total_rows)

    # Write Parquet if absent ---------------------------------------------
    if not parquet_dir.exists():
        parquet_dir.mkdir(parents=True, exist_ok=True)
        n_chunks = math.ceil(total_rows / chunk_rows)
        logger.info("Writing %d chunk(s) → %s", n_chunks, parquet_dir)
        for c in range(n_chunks):
            start, end = c * chunk_rows, min((c + 1) * chunk_rows, total_rows)
            ids_slice = (
                id_values[start:end]
                if id_values is not None
                else np.arange(start, end, dtype=np.int64)
            )
            df_dict: Dict[str, object] = {ID_COL: ids_slice}
            for s in column_specs:
                df_dict[s.name] = s.slice_chunk(total_rows, start, end)
            df = pd.DataFrame(df_dict, copy=False)
            pq.write_table(
                pa.Table.from_pandas(df, preserve_index=False),
                parquet_dir / f"chunk_{c:05d}.parquet",
                compression=parquet_codec,
                # id → no dictionary encoding; strings honour spec setting
                use_dictionary={ID_COL: False, **{s.name: s.use_dictionary for s in column_specs}},
                write_statistics=False,
                data_page_size=1 << 20,
            )
    else:
        logger.info("Parquet dir exists – reuse %s", parquet_dir)

    # Load into DuckDB -----------------------------------------------------
    con = duckdb.connect(duckdb_path)
    con.execute("PRAGMA force_compression='dictionary'")
    table_present = bool(
        con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
    )

    if table_present and if_exists == "error":
        raise RuntimeError(f"Table '{table_name}' already exists in {duckdb_path}")
    if table_present and if_exists == "replace":
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        table_present = False

    if not table_present:
        cols = ", ".join([ID_COL] + [s.name for s in column_specs])
        con.execute(
            f"CREATE TABLE {table_name} AS SELECT {cols} FROM read_parquet('{parquet_dir}/*.parquet')"
        )
        con.execute("CHECKPOINT")
    else:
        logger.info("Skip table creation (already present)")

    con.close()

    if cleanup:
        shutil.rmtree(parquet_dir, ignore_errors=True)
    return duckdb_path


# ── Helper: build many tables into one DB ─────────────────────────────────--


def build_db_with_multiple_tables(
    *,
    duckdb_path: Union[str, Path],
    variants: Sequence[tuple[str, List[ColumnSpec]]],
    total_rows: int,
    id_values_map: Optional[Dict[str, np.ndarray]] = None,
    chunk_rows: int = 1_000_000,
    parquet_codec: str = "zstd",
    seed_base: int = 42,
) -> Path:
    """Generate/append each variant; optional explicit ID arrays per table."""

    for idx, (tbl_name, specs) in enumerate(variants, start=1):
        generate_string_benchmark(
            duckdb_path=duckdb_path,
            table_name=tbl_name,
            if_exists="skip",
            total_rows=total_rows,
            column_specs=specs,
            id_values=None if id_values_map is None else id_values_map[tbl_name],
            chunk_rows=chunk_rows,
            parquet_codec=parquet_codec,
            seed=seed_base + idx,
        )
    return Path(duckdb_path)

##############################################################################
# ░░ Part 2 – Two‑table benchmark driver ░░                                  #
##############################################################################

# Optional repo root injection
root_directory = Path(__file__).resolve().parents[1]
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

# Systems & runner -----------------------------------------------------------
from config.systems.duckdb import (
    DUCK_DB_MAIN,
    UnifiedStringDictionary_lock_free_16mB,
    UnifiedStringDictionary_lock_free_512K,
)
from src.runner.experiment_runner import run  # type: ignore

# get_data_path helper -------------------------------------------------------

from src.utils import get_data_path  # type: ignore


def get_data_path(rel: str) -> str:  # noqa: D401 – helper
    return str(Path(".") / rel)

# Grid parameters ------------------------------------------------------------
LengthSpec = Union[int, Tuple[int, int]]
LENGTH_SPECS:    Sequence[LengthSpec] = [32]
TOTAL_ROWS_LIST: Sequence[int]        = [5_000]
N_UNIQUE_LIST:   Sequence[int]        = [1000]
S_VALUES:        Sequence[float]      = [0.0]

# >>>>>>>>>> SET THE DEFAULT JOIN CARDINALITY IN ONE PLACE <<<<<<<<<<<<<<
EXPECTED_JOIN_ROWS_DEFAULT: Optional[int] = 10_000_000  # e.g. 100_000 or None for all rows
# -----------------------------------------------------------------------

CHUNK_ROWS = 1_000_000
PARQUET_CODEC = "zstd"
SEED_BASE = 999

TABLE_A, TABLE_B = "varchars_a", "varchars_b"

# Queries --------------------------------------------------------------------
CUSTOM_QUERIES: List[Query] = [
    {
        "name": "id_equality_join",
        "index": 0,
        "run_script": {
            "duckdb": f"SELECT COUNT(*) AS cnt FROM {TABLE_A} AS a JOIN {TABLE_B} AS b USING({ID_COL})"
        },
    },
]

# Helper functions -----------------------------------------------------------

def len_key(spec: LengthSpec) -> str:
    return str(spec) if isinstance(spec, int) else f"{spec[0]}-{spec[1]}"


def build_db_path(len_spec: LengthSpec, n_unique: int, s_val: float, root: Optional[Path] = None) -> Path:
    dist_dir = f"varchars_grp_size_zipf{s_val}"
    len_dir = f"len_{len_key(len_spec)}"
    fname = f"varchars-grp-size-{n_unique}.db"
    rel = os.path.join(dist_dir, len_dir, fname)
    return Path(get_data_path(rel) if root is None else root / rel)


def make_column_specs(spec: LengthSpec, n_unique: int, s_val: float) -> List[ColumnSpec]:
    return [
        ColumnSpec("str1", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),
        ColumnSpec("str2", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),
        ColumnSpec("str3", n_unique, spec, "zipf", zipf_s=s_val, use_dictionary=True),
    ]


# Assemble datasets ----------------------------------------------------------

def assemble_datasets(*, root: Optional[Path] = None, join_rows_override: Optional[int] = None) -> List[DataSet]:
    datasets: List[DataSet] = []

    for idx, (len_spec, rows, uniques, s_val) in enumerate(
        (x for x in __import__("itertools").product(LENGTH_SPECS, TOTAL_ROWS_LIST, N_UNIQUE_LIST, S_VALUES)),
        start=1,
    ):
        # Determine how many IDs should overlap between the two tables
        join_rows = (
            min(join_rows_override, rows) if join_rows_override is not None else rows
        )
        if join_rows <= 0:
            raise ValueError("join_rows must be positive")

        db_path = build_db_path(len_spec, uniques, s_val, root)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Generate matching, non‑sequential ID pools ────────────────────
        rng = np.random.default_rng(SEED_BASE + idx)
        id_pool_size = rows * 10  # plenty of headroom for uniqueness
        overlap_ids = rng.choice(id_pool_size, size=join_rows, replace=False)  # shared

        remaining_pool = np.setdiff1d(np.arange(id_pool_size), overlap_ids, assume_unique=True)
        ids_a_extra = (
            rng.choice(remaining_pool, size=rows - join_rows, replace=False)
            if rows > join_rows
            else np.empty(0, dtype=np.int64)
        )
        remaining_pool_for_b = np.setdiff1d(remaining_pool, ids_a_extra, assume_unique=True)
        ids_b_extra = (
            rng.choice(remaining_pool_for_b, size=rows - join_rows, replace=False)
            if rows > join_rows
            else np.empty(0, dtype=np.int64)
        )

        ids_a = np.concatenate([overlap_ids, ids_a_extra]).astype(np.int64)
        ids_b = np.concatenate([overlap_ids, ids_b_extra]).astype(np.int64)
        rng.shuffle(ids_a)
        rng.shuffle(ids_b)

        id_values_map = {TABLE_A: ids_a, TABLE_B: ids_b}

        # ── Generate both tables (skip if already present) ─────────────────
        build_db_with_multiple_tables(
            duckdb_path=db_path,
            variants=[
                (TABLE_A, make_column_specs(len_spec, uniques, s_val)),
                (TABLE_B, make_column_specs(len_spec, uniques, s_val)),
            ],
            total_rows=rows,
            id_values_map=id_values_map,
            chunk_rows=CHUNK_ROWS,
            parquet_codec=PARQUET_CODEC,
            seed_base=SEED_BASE + idx * 10,
        )

        datasets.append(
            {
                "name": f"len{len_key(len_spec)}_uni{uniques}_zipf{s_val}",
                "setup_script": {
                    "duckdb": f"ATTACH '{db_path}' (READ_ONLY); USE '{db_path.stem}';"
                },
                "config": {
                    "string_length": len_spec,
                    "n_unique": uniques,
                    "zipf_s": s_val,
                    "tables": [TABLE_A, TABLE_B],
                    "expected_join_rows": join_rows,
                },
            }
        )
    return datasets


# Build benchmark ------------------------------------------------------------

def build_benchmark(*, root: Optional[Path] = None, join_rows_override: Optional[int] = None) -> Benchmark:
    return {
        "name": "string_benchmark_zipf_grid_two_tables_with_id_pool",
        "datasets": assemble_datasets(root=root, join_rows_override=join_rows_override),
        "queries": CUSTOM_QUERIES,
    }


# Runtime settings -----------------------------------------------------------
RUN_SETTINGS = {"n_parallel": 1, "n_runs": 6}
SYSTEM_SETTINGS = [{"n_threads": 1}]
SYSTEMS = [DUCK_DB_MAIN, UnifiedStringDictionary_lock_free_16mB]

##############################################################################
# ░░ Runtime control ░░                                                      #
##############################################################################

# Folder where the benchmark DBs will be created (None → default repo path)
ROOT_OVERRIDE: Optional[Path] = None

# Flip this to False if you only want to build the data and *not* launch the
# runner.  Handy when the experiment runner isn't available in the environment.
RUN_BENCHMARK: bool = True

# If you wish to override the default join cardinality for *all* datasets,
# set it here (None → fall back to EXPECTED_JOIN_ROWS_DEFAULT defined above).
JOIN_ROWS_OVERRIDE: Optional[int] = None

##############################################################################
# ░░ Main – entry point without CLI ░░                                       #
##############################################################################

def main() -> None:
    logger.info("Starting dataset generation (CLI‑less mode)…")
    benchmark = build_benchmark(
        root=ROOT_OVERRIDE, join_rows_override=JOIN_ROWS_OVERRIDE or EXPECTED_JOIN_ROWS_DEFAULT
    )

    if not RUN_BENCHMARK:
        logger.info("RUN_BENCHMARK = False → data generation complete; skipping runner.")
        return

    cfg: RunConfig = {
        "name": "USSR_vs_baseline_zipf_grid_two_tables_with_id_pool",
        "run_settings": RUN_SETTINGS,
        "system_settings": SYSTEM_SETTINGS,
        "systems": SYSTEMS,
        "benchmarks": benchmark,
    }

    logger.info("Launching experiment runner…")
    run(cfg)


if __name__ == "__main__":
    main()

##############################################################################
# ░░ Public re‑exports ░░                                                    #
##############################################################################

__all__ = [
    "ColumnSpec",
    "generate_string_benchmark",
    "build_db_with_multiple_tables",
]
