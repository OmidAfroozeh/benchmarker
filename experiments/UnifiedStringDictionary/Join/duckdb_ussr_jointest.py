
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
except Exception:  # pragma: no cover – standalone mode

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


DistributionLiteral = Literal["uniform", "zipf"]


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


# ── Top‑level generator ─────────────────────────────────────────────────────

ID_COLUMN = "id"  # public constant so downstream code can reference the column name

def generate_string_benchmark(
    *,
    duckdb_path: Union[str, Path],
    total_rows: int,
    column_specs: List[ColumnSpec],
    table_name: str = "varchars",
    if_exists: Literal["skip", "replace", "error"] = "skip",
    chunk_rows: int = 5_000_000,
    parquet_codec: str = "zstd",
    alphabet: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    cleanup: bool = True,
    global_unique: bool = False,
) -> Path:
    """Create (or append) a Parquet‑backed varchar table inside *duckdb_path*.

    Every row gets a unique sequential integer in the :pydata:`ID_COLUMN`
    column, starting at 0. The sequence is deterministic and independent of
    the RNG seed so that identically‑shaped tables (e.g. *A* & *B*) share the
    same *id*s and can be joined via *id* equality.
    """

    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if not column_specs:
        raise ValueError("Need at least one ColumnSpec")

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

            # Build chunk DataFrame with the mandatory `id` column ----------
            data: Dict[str, Union[np.ndarray, range]] = {
                ID_COLUMN: np.arange(start, end, dtype="int64"),
            }
            for s in column_specs:
                data[s.name] = s.slice_chunk(total_rows, start, end)

            df = pd.DataFrame(data, copy=False)

            pq.write_table(
                pa.Table.from_pandas(df, preserve_index=False),
                parquet_dir / f"chunk_{c:05d}.parquet",
                compression=parquet_codec,
                use_dictionary={s.name: s.use_dictionary for s in column_specs},
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
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table_name]
        ).fetchone()
    )

    if table_present and if_exists == "error":
        raise RuntimeError(f"Table '{table_name}' already exists in {duckdb_path}")
    if table_present and if_exists == "replace":
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        table_present = False

    if not table_present:
        cols = ", ".join([ID_COLUMN] + [s.name for s in column_specs])
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


# ── Helper: build many tables into one DB ───────────────────────────────────

def build_db_with_multiple_tables(
    *,
    duckdb_path: Union[str, Path],
    variants: Sequence[tuple[str, List[ColumnSpec]]],
    total_rows: int,
    chunk_rows: int = 1_000_000,
    parquet_codec: str = "zstd",
    seed_base: int = 42,
) -> Path:
    for idx, (tbl_name, specs) in enumerate(variants, start=1):
        generate_string_benchmark(
            duckdb_path=duckdb_path,
            table_name=tbl_name,
            if_exists="skip",
            total_rows=total_rows,
            column_specs=specs,
            chunk_rows=chunk_rows,
            parquet_codec=parquet_codec,
            seed=seed_base + idx,
        )
    return Path(duckdb_path)

# Optional repo root injection
root_directory = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_directory))
grandparent = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(grandparent))
great_grandparent = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(great_grandparent))

# Systems & runner -----------------------------------------------------------
from config.systems.duckdb import (
    DUCK_DB_MAIN,
    USSR_SALT_CLEAN,
    Unified_String_Dictionary,
    unified_string_dictionary_sort_test
)  # type: ignore
from src.runner.experiment_runner import run  # type: ignore

# get_data_path helper -------------------------------------------------------
try:
    from src.utils import get_data_path  # type: ignore
except Exception:

    def get_data_path(rel: str) -> str:  # noqa: D401 – helper
        return str(Path("/mnt/benchmarks") / rel)

# Grid parameters ------------------------------------------------------------
LengthSpec = Union[int, Tuple[int, int]]
LENGTH_SPECS: Sequence[LengthSpec] = [32]
TOTAL_ROWS_LIST: Sequence[int] = [10_000_000]
N_UNIQUE_LIST: Sequence[int] = [100]
S_VALUES: Sequence[float] = [0.0]

CHUNK_ROWS = 1_000_000
PARQUET_CODEC = "zstd"
SEED_BASE = 999

TABLE_A, TABLE_B = "varchars_a", "varchars_b"

# Queries --------------------------------------------------------------------
CUSTOM_QUERIES: List[Query] = [
    # {
    #     "name": "join_on_integer_keys",
    #     "index": 0,
    #     "run_script": {
    #         "duckdb": f"select * from varchars_a join varchars_b on varchars_a.id = varchars_b.id;"
    #     },
    # },
    # {
    #     "name": "join_on_integer_keys_limited_result",
    #     "index": 1,
    #     "run_script": {
    #         "duckdb": f"select * from varchars_a join varchars_b on varchars_a.id = varchars_b.id limit 10;"
    #     },
    # },
    # {
    #     "name": "sort_on_integer_keys_result",
    #     "index": 2,
    #     "run_script": {
    #         "duckdb": f"select * from varchars_a order by id;"
    #     },
    # },
    # {
    #     "name": "sort_on_integer_keys_limited_result",
    #     "index": 3,
    #     "run_script": {
    #         "duckdb": f"select * from varchars_a order by id limit 10;"
    #     },
    # },
    {
        "name": "aggregate_after_join",
        "index": 0,
        "run_script": {
            "duckdb": f"select varchars_a.str1 from varchars_a join varchars_b on varchars_a.id = varchars_b.id group by varchars_a.str1;"
        },
    },
]

# Helper functions -----------------------------------------------------------

def len_key(spec: LengthSpec) -> str:
    return str(spec) if isinstance(spec, int) else f"{spec[0]}-{spec[1]}"


def build_db_path(len_spec: LengthSpec, n_unique: int, s_val: float, root: Optional[Path] = None) -> Path:
    dist_dir = f"varchars_grp_size_zipf_join_{s_val}"
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

def assemble_datasets(root: Optional[Path] = None) -> List[DataSet]:
    datasets: List[DataSet] = []
    for idx, (len_spec, rows, uniques, s_val) in enumerate(
        (
            x
            for x in __import__("itertools").product(
                LENGTH_SPECS, TOTAL_ROWS_LIST, N_UNIQUE_LIST, S_VALUES
            )
        ),
        start=1,
    ):
        db_path = build_db_path(len_spec, uniques, s_val, root)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # generate both tables (skip if present)
        build_db_with_multiple_tables(
            duckdb_path=db_path,
            variants=[
                (TABLE_A, make_column_specs(len_spec, uniques, s_val)),
                (TABLE_B, make_column_specs(len_spec, uniques, s_val)),
            ],
            total_rows=rows,
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
                },
            }
        )
    return datasets


# Build benchmark ------------------------------------------------------------

def build_benchmark(root: Optional[Path] = None) -> Benchmark:
    return {
        "name": "string_benchmark_zipf_grid_two_tables",
        "datasets": assemble_datasets(root),
        "queries": CUSTOM_QUERIES,
    }


# Runtime settings -----------------------------------------------------------
RUN_SETTINGS = {"n_parallel": 1, "n_runs": 6}
SYSTEM_SETTINGS = [{"n_threads": 8}]
SYSTEMS = [DUCK_DB_MAIN, Unified_String_Dictionary, unified_string_dictionary_sort_test]


# Main entry point -----------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Generate two‑table varchar benchmarks (with row IDs) and optionally run them"
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override benchmark base folder (defaults to get_data_path)",
    )
    p.add_argument("--norun", action="store_true", help="Only generate data – do not launch runner")
    args = p.parse_args()

    benchmark = build_benchmark(args.root)

    cfg: RunConfig = {
        "name": "USSR_vs_baseline_zipf_grid_two_tables",
        "run_settings": RUN_SETTINGS,
        "system_settings": SYSTEM_SETTINGS,
        "systems": SYSTEMS,
        "benchmarks": benchmark,
    }

    if args.norun:
        logger.info("Data generation complete – skipping run (\\--norun)\nConfig: %s", cfg)
    else:
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
    "ID_COLUMN",
]
