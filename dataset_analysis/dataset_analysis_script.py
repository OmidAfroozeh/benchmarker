# analyze_duckdbs.py
"""
Walk a directory tree, locate all DuckDB files (default ``*.db``), and collect
per‑table/per‑column statistics plus storage‑segment metadata, **without any
CLI arguments**.  Just edit the two constants below and run:

```bash
python analyze_duckdbs.py
```

* ``ROOT_DIR``   – root folder to begin the recursive scan (edit to suit).
* ``FILE_GLOB``  – filename pattern for DuckDB files (default: ``"*.db"``).

Outputs (in the working directory)
----------------------------------
* ``column_statistics.csv`` – one row per *database × table × column*.
* ``storage_summary.csv``   – one row per *database × table × column × codec*.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import List, Tuple

import duckdb
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG –‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑  Edit the path(s) below  ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path("/Users/omid/PycharmProjects/benchmarker_omid/_output/data").expanduser()
FILE_GLOB = "*.db"               # e.g. "*.duckdb" if you use that extension
# ──────────────────────────────────────────────────────────────────────────────

# Helpers ─────────────────────────────────────────────────────────────────────

def _list_tables(con: duckdb.DuckDBPyConnection) -> List[str]:
    return (
        con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND table_type = 'BASE TABLE'
            """
        )
        .fetch_df()["table_name"].tolist()
    )


def _collect_column_stats(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    cols = con.execute(f"PRAGMA table_info('{table}')").df()["name"].tolist()
    frames = []
    for col in cols:
        frames.append(
            con.execute(
                f"""
                SELECT '{col}' AS column,
                       COUNT(DISTINCT {col}) AS unique_count,
                       AVG(LENGTH({col}))    AS avg_length,
                       MIN(LENGTH({col}))    AS min_length,
                       MAX(LENGTH({col}))    AS max_length
                FROM {table}
                WHERE {col} IS NOT NULL
                """
            ).fetch_df()
        )
    return pd.concat(frames, ignore_index=True)


def _collect_storage_summary(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    df = con.execute(
        f"SELECT * FROM pragma_storage_info('{table}') WHERE segment_type = 'VARCHAR'"
    ).fetch_df()
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["column_name", "compression"], as_index=False)
        .agg(total_count=("count", "sum"), num_segments=("count", "count"), avg_count=("count", "mean"))
        .rename(columns={"column_name": "column"})
    )


# Core collectors ─────────────────────────────────────────────────────────────

def _process_database(db_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(str(db_path), read_only=True)
    col_frames: List[pd.DataFrame] = []
    stor_frames: List[pd.DataFrame] = []

    try:
        tables = _list_tables(con)
    except duckdb.Error:
        print(f"[WARN] Could not read {db_path}; skipping.")
        return pd.DataFrame(), pd.DataFrame()

    for tbl in tables:
        num_rows = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        num_cols = len(con.execute(f"PRAGMA table_info('{tbl}')").fetchall())

        cs = _collect_column_stats(con, tbl)
        cs.insert(0, "database", str(db_path))
        cs.insert(1, "table", tbl)
        cs.insert(2, "num_rows", num_rows)
        cs.insert(3, "num_columns", num_cols)
        col_frames.append(cs)

        ss = _collect_storage_summary(con, tbl)
        if not ss.empty:
            ss.insert(0, "database", str(db_path))
            ss.insert(1, "table", tbl)
            stor_frames.append(ss)

    cols_df = pd.concat(col_frames, ignore_index=True) if col_frames else pd.DataFrame()
    stor_df = pd.concat(stor_frames, ignore_index=True) if stor_frames else pd.DataFrame()
    return cols_df, stor_df


def _walk_and_process(root: Path, pattern: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_cols: List[pd.DataFrame] = []
    all_stor: List[pd.DataFrame] = []

    for path in root.rglob("*"):
        if path.is_file() and fnmatch.fnmatch(path.name, pattern):
            print(f"[+] Processing {path}")
            c_df, s_df = _process_database(path)
            if not c_df.empty:
                all_cols.append(c_df)
            if not s_df.empty:
                all_stor.append(s_df)

    cols = pd.concat(all_cols, ignore_index=True) if all_cols else pd.DataFrame()
    stor = pd.concat(all_stor, ignore_index=True) if all_stor else pd.DataFrame()
    return cols, stor


# Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {ROOT_DIR}")

    col_stats, stor_stats = _walk_and_process(ROOT_DIR, FILE_GLOB)

    if col_stats.empty and stor_stats.empty:
        print("No DuckDB files or user tables found – nothing to write.")
        return

    script_dir = Path(__file__).parent.resolve()
    col_stats_path = script_dir / "column_statistics.csv"
    stor_stats_path = script_dir / "storage_summary.csv"

    if not col_stats.empty:
        col_stats.to_csv(col_stats_path, index=False)
        print(f"[✓] {col_stats_path.name} written.")
    if not stor_stats.empty:
        stor_stats.to_csv(stor_stats_path, index=False)
        print(f"[✓] {stor_stats_path.name} written.")


if __name__ == "__main__":
    main()
