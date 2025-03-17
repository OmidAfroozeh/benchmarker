# import os
# from typing import List
#
# import duckdb
#
# from src.logger import get_logger
# from src.models import DataSet, Benchmark, Query
# from src.utils import get_data_path, pad
#
# logger = get_logger(__name__)
#
# MICRO_BENCHMARK_QUERY: List[Query] = [
#     {
#         'name': 'single_column_groupby',
#         'index': 0,
#         'run_script': {
#             "duckdb": "select str1 from varchars group by str1",
#         }
#     },
#     {
#         'name': 'double_column_groupby',
#         'index': 1,
#         'run_script': {
#             "duckdb": "select str1, str2 from varchars group by str1, str2",
#         }
#     },
# ]
#
# def get_string_len_benchmark(string_len: List[int]) -> Benchmark:
#
#     datasets: List[DataSet] = __generate_and_return_string_microbenchmark_data(string_len)
#
#     queries = MICRO_BENCHMARK_QUERY
#
#     return {
#         'name': 'string_micro_benchmark',
#         'datasets': datasets,
#         'queries': queries
#     }
#
#
# def __get_tpcds_file_path(sf: int) -> str:
#     file_name =  os.path.join('varchars_grp_size_uniform', f'varchars-grp-size-{sf}.db')
#     return get_data_path(file_name)
#
#
# def __generate_and_return_string_microbenchmark_data(str_lens: List[int]) -> List[DataSet]:
#     __generate_string_microbenchmark_data(str_lens)
#
#     datasets: List[DataSet] = []
#     for sf in str_lens:
#         duckdb_file_path = __get_tpcds_file_path(sf)
#         duckdb_file_name_without_extension = os.path.splitext(os.path.basename(duckdb_file_path))[0]
#
#         setup_script = {
#             'duckdb': f"ATTACH '{duckdb_file_path}' (READ_ONLY); USE '{duckdb_file_name_without_extension}';"
#         }
#
#         dataset: DataSet = {
#             'name': f'tpcds-{sf}',
#             'setup_script': setup_script,
#             'config': {
#                 'group_size': sf
#             }
#         }
#
#         datasets.append(dataset)
#
#     return datasets
#
#
# # def __generate_string_microbenchmark_data(str_len: List[int]):
# #     for (index, sl) in enumerate(str_len):
# #         logger.info(f'Generating data for string micro benchmark {sl} ({index + 1}/{len(str_len)}) ...')
# #         duckdb_file_path = __get_tpcds_file_path(sl)
# #
# #         # Only generate the data if the file does not exist
# #         if os.path.exists(duckdb_file_path):
# #             logger.info(f'File {duckdb_file_path} already exists, skipping...')
# #             continue
# #         else:
# #             logger.info(f'File {duckdb_file_path} does not exist, generating...')
# #
# #         logger.info(f'Started to generate data for string microbenchmark string size {sl} ...')
# #
# #         TOTAL_ROWS = 1000_000_000
# #         div = TOTAL_ROWS // sl
# #
# #         con = duckdb.connect(duckdb_file_path)
# #         query_tpcds = f"""
# #             PRAGMA force_compression='dictionary';
# #             CREATE TABLE varchars AS SELECT concat('thisisarandomstringjusttoseeblahblahthisisarandomstringjusttosee', i//{div}) AS str1, concat('thisisarandomstringjusttoseeblahblahthisisarandomstringjusttosee', i//{div}) AS str2 FROM range(100_000_000) tbl(i);
# #             checkpoint;
# #         """
# #         con.sql(query_tpcds)
# #         con.close()
#
# # ── NEW DEPENDENCIES ──────────────────────────────────────────────
# import math, string
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import pyarrow as pa, pyarrow.parquet as pq
# # -----------------------------------------------------------------
#
#
#
# def __generate_string_microbenchmark_data(unique_counts: List[int]):
#     """
#     For each value in `unique_counts`:
#       • generate a Parquet data-set with that many distinct strings
#         (uniformly repeated, no skew, 100M rows, 64-char strings, 5M-row chunks);
#       • create 1 DuckDB database containing a dictionary-compressed table
#         `varchars` loaded from the Parquet files.
#     Skip generation if the .db already exists.
#     """
#
#     # --------------------- config --------------------- #
#     TOTAL_ROWS    = 10_000_000
#     STR_LEN       = 64
#     CHUNK_ROWS    = 5_000_000
#     PARQUET_CODEC = "zstd"
#     # -------------------------------------------------- #
#
#     alphabet = np.array(list(string.ascii_lowercase), dtype="U1")
#
#     def _string_pool(n_unique: int) -> np.ndarray:
#         chars = np.random.choice(alphabet, size=(n_unique, STR_LEN))
#         return np.asarray(["".join(row) for row in chars], dtype="O")
#
#     # ---------------- main loop ----------------------- #
#     for idx, n_unique in enumerate(unique_counts, start=1):
#         logger.info(
#             f"Generating uniform data-set {idx}/{len(unique_counts)} "
#             f"— {n_unique:,} uniques ..."
#         )
#
#         duckdb_path = __get_tpcds_file_path(n_unique)
#         if os.path.exists(duckdb_path):
#             logger.info("  DuckDB file exists → skip.")
#             continue
#
#         parquet_dir = Path(duckdb_path).with_suffix("").with_name(
#             Path(duckdb_path).stem + "_parquet"
#         )
#         parquet_dir.mkdir(parents=True, exist_ok=True)
#
#         # --- prepare uniform string values ---
#         rows_per_string = TOTAL_ROWS // n_unique
#         pool = _string_pool(n_unique)
#         col = np.repeat(pool, rows_per_string)
#
#         # In case of remainder, append extra values from the beginning
#         remainder = TOTAL_ROWS - len(col)
#         if remainder > 0:
#             col = np.concatenate([col, pool[:remainder]])
#
#         assert len(col) == TOTAL_ROWS, "Total rows must match expected size"
#
#         # Shuffle for randomness
#         np.random.shuffle(col)
#
#         # Write Parquet chunks
#         n_chunks = math.ceil(TOTAL_ROWS / CHUNK_ROWS)
#         for c in range(n_chunks):
#             start = c * CHUNK_ROWS
#             end = min(start + CHUNK_ROWS, TOTAL_ROWS)
#             df = pd.DataFrame({"str1": col[start:end], "str2": col[start:end]}, copy=False)
#
#             pq.write_table(
#                 pa.Table.from_pandas(df, preserve_index=False),
#                 parquet_dir / f"chunk_{c:05d}.parquet",
#                 compression=PARQUET_CODEC,
#                 use_dictionary=True,
#                 write_statistics=False,
#                 data_page_size=1 << 20,
#             )
#         logger.info("  Parquet data-set ready → %s", parquet_dir)
#
#         # Create DuckDB database
#         con = duckdb.connect(duckdb_path)
#         con.execute("PRAGMA force_compression='dictionary'")
#         con.execute(
#             f"CREATE TABLE varchars AS "
#             f"SELECT * FROM read_parquet('{parquet_dir}/*.parquet')"
#         )
#         con.execute("CHECKPOINT")
#         con.close()
#         logger.info("  DuckDB file ready → %s", duckdb_path)
