import os
from typing import List

import duckdb

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path, pad

# ── NEW DEPENDENCIES ──────────────────────────────────────────────
import math, string
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
# -----------------------------------------------------------------

logger = get_logger(__name__)

MICRO_BENCHMARK_QUERY: List[Query] = [
    {
        'name': 'double_column_groupby',
        'index': 1,
        'run_script': {
            "duckdb": "select str1, str2 from varchars group by str1, str2",
        }
    }
]

def get_string_len_benchmark(string_len: List[int]) -> Benchmark:

    datasets: List[DataSet] = __generate_and_return_string_microbenchmark_data(string_len)

    queries = MICRO_BENCHMARK_QUERY

    return {
        'name': 'string_micro_benchmark',
        'datasets': datasets,
        'queries': queries
    }


def __get_tpcds_file_path(sf: int) -> str:
    file_name =  os.path.join('meassure', f'overhead-{sf}.db')
    return get_data_path(file_name)


def __generate_and_return_string_microbenchmark_data(str_lens: List[int]) -> List[DataSet]:
    __generate_string_microbenchmark_data(str_lens)

    datasets: List[DataSet] = []
    for sf in str_lens:
        duckdb_file_path = __get_tpcds_file_path(sf)
        duckdb_file_name_without_extension = os.path.splitext(os.path.basename(duckdb_file_path))[0]

        setup_script = {
            'duckdb': f"ATTACH '{duckdb_file_path}' (READ_ONLY); USE '{duckdb_file_name_without_extension}';"
        }

        dataset: DataSet = {
            'name': f'tpcds-{sf}',
            'setup_script': setup_script,
            'config': {
                'string_length': sf
            }
        }

        datasets.append(dataset)

    return datasets


def __generate_string_microbenchmark_data(str_len: List[int]):
    for (index, sl) in enumerate(str_len):
        logger.info(f'Generating data for string micro benchmark {sl} ({index + 1}/{len(str_len)}) ...')
        duckdb_file_path = __get_tpcds_file_path(sl)

        # Only generate the data if the file does not exist
        if os.path.exists(duckdb_file_path):
            logger.info(f'File {duckdb_file_path} already exists, skipping...')
            continue
        else:
            logger.info(f'File {duckdb_file_path} does not exist, generating...')

        logger.info(f'Started to generate data for string microbenchmark string size {sl} ...')

        # Generate temp_str of length sl
        base_str = "abds123412341234"

        con = duckdb.connect(duckdb_file_path)
        query_tpcds = f"""
            CREATE TABLE varchars AS SELECT concat('{base_str}', i) AS str1, concat('{base_str}', i) AS str2 FROM range({sl}) tbl(i);
        """
        con.sql(query_tpcds)
        con.close()


