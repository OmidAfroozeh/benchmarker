import os
from typing import List

import duckdb

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path, pad

logger = get_logger(__name__)

TPC_DS_QUERIES: List[Query] = [
    {
        'name': f'clickbench',
        'run_script': {
            "duckdb": "",
        }
    }
]
