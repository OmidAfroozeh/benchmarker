import os
from typing import List

import duckdb

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path, pad

logger = get_logger(__name__)

TPC_H_QUERIES: List[Query] = [
    # {
    #     'name': f'tpch{i + 1}',
    #     'index': i,
    #     'run_script': {
    #         "duckdb": f"PRAGMA tpch({i + 1});",
    #     }
    # } for i in range(22)
    {
        'name': f'tpch_varchar_test_count_star',
        'index': 0,
        'run_script': {
            "duckdb": f"select count(*) from nation JOIN customer  on nation.n_nation_uuid_str = customer.n_nation_uuid_str;",
        }
    },
    {
        'name': f'tpch_varchar_test_star',
        'index': 1,
        'run_script': {
            "duckdb": f"select * from nation JOIN customer  on nation.n_nation_uuid_str = customer.n_nation_uuid_str;",
        }
    }
]


def get_tpch_benchmark_str_keys(scale_factors: List[int]) -> Benchmark:
    datasets: List[DataSet] = __generate_and_return_tpc_data(scale_factors)

    queries = TPC_H_QUERIES

    return {
        'name': 'tpch',
        'datasets': datasets,
        'queries': queries
    }


def __get_tpch_file_path(sf: int) -> str:
    file_name = os.path.join('tpch-string-keys', f'tpch-sf{sf}.db')
    return get_data_path(file_name)


def __generate_and_return_tpc_data(sfs: List[int]) -> List[DataSet]:
    __generate_tpch_data(sfs)

    datasets: List[DataSet] = []
    for sf in sfs:
        duckdb_file_path = __get_tpch_file_path(sf)
        duckdb_file_name_without_extension = os.path.splitext(os.path.basename(duckdb_file_path))[0]

        setup_script = {
            'duckdb': f"ATTACH '{duckdb_file_path}' (READ_ONLY); USE '{duckdb_file_name_without_extension}'; PRAGMA disable_progress_bar;"
        }

        dataset: DataSet = {
            'name': f'tpch-{sf}',
            'setup_script': setup_script,
            'config': {
                'sf': sf
            }
        }

        datasets.append(dataset)

    return datasets


def __generate_tpch_data(sfs: List[int]):
    for (index, sf) in enumerate(sfs):
        logger.info(f'Generating data for TPC-H scale factor {sf} ({index + 1}/{len(sfs)}) ...')
        duckdb_file_path = __get_tpch_file_path(sf)
        # only generate the data if the file does not exist
        if os.path.exists(duckdb_file_path):
            logger.info(f'File {duckdb_file_path} already exists, skipping...')
            continue
        else:
            logger.info(f'File {duckdb_file_path} does not exist, generating...')

        logger.info(f'Started to generate data for TPC-H scale factor {sf} ...')

        con = duckdb.connect(duckdb_file_path)
        query_tpch = f"""
            INSTALL tpch;
            LOAD tpch;
            CALL dbgen(sf = {sf});

            ALTER TABLE customer ADD COLUMN custkey_uuid VARCHAR;
            UPDATE customer SET custkey_uuid = uuid();

            ALTER TABLE orders ADD COLUMN o_custkey_uuid VARCHAR;
            UPDATE orders SET o_custkey_uuid = (
                SELECT c.custkey_uuid
                FROM customer c
                WHERE c.c_custkey = orders.o_custkey
            );

            ALTER TABLE orders ADD COLUMN orderkey_uuid VARCHAR;
            UPDATE orders SET orderkey_uuid = uuid();

            ALTER TABLE lineitem ADD COLUMN l_orderkey_uuid VARCHAR;
            UPDATE lineitem SET l_orderkey_uuid = (
                SELECT o.orderkey_uuid
                FROM orders o
                WHERE o.o_orderkey = lineitem.l_orderkey
            );

            ALTER TABLE part ADD COLUMN partkey_uuid VARCHAR;
            UPDATE part SET partkey_uuid = uuid();

            ALTER TABLE supplier ADD COLUMN suppkey_uuid VARCHAR;
            UPDATE supplier SET suppkey_uuid = uuid();

            ALTER TABLE lineitem ADD COLUMN l_partkey_uuid VARCHAR;
            ALTER TABLE lineitem ADD COLUMN l_suppkey_uuid VARCHAR;
            UPDATE lineitem SET l_partkey_uuid = (
                SELECT p.partkey_uuid
                FROM part p
                WHERE p.p_partkey = lineitem.l_partkey
            );
            UPDATE lineitem SET l_suppkey_uuid = (
                SELECT s.suppkey_uuid
                FROM supplier s
                WHERE s.s_suppkey = lineitem.l_suppkey
            );

            ALTER TABLE partsupp ADD COLUMN ps_partkey_uuid VARCHAR;
            ALTER TABLE partsupp ADD COLUMN ps_suppkey_uuid VARCHAR;
            UPDATE partsupp SET ps_partkey_uuid = (
                SELECT p.partkey_uuid
                FROM part p
                WHERE p.p_partkey = partsupp.ps_partkey
            );
            UPDATE partsupp SET ps_suppkey_uuid = (
                SELECT s.suppkey_uuid
                FROM supplier s
                WHERE s.s_suppkey = partsupp.ps_suppkey
            );

            ALTER TABLE nation ADD COLUMN nationkey_uuid VARCHAR;
            UPDATE nation SET nationkey_uuid = uuid();

            ALTER TABLE region ADD COLUMN regionkey_uuid VARCHAR;
            UPDATE region SET regionkey_uuid = uuid();

            ALTER TABLE customer ADD COLUMN c_nationkey_uuid VARCHAR;
            UPDATE customer SET c_nationkey_uuid = (
                SELECT n.nationkey_uuid
                FROM nation n
                WHERE n.n_nationkey = customer.c_nationkey
            );

            ALTER TABLE supplier ADD COLUMN s_nationkey_uuid VARCHAR;
            UPDATE supplier SET s_nationkey_uuid = (
                SELECT n.nationkey_uuid
                FROM nation n
                WHERE n.n_nationkey = supplier.s_nationkey
            );

            ALTER TABLE nation ADD COLUMN n_regionkey_uuid VARCHAR;
            UPDATE nation SET n_regionkey_uuid = (
                SELECT r.regionkey_uuid
                FROM region r
                WHERE r.r_regionkey = nation.n_regionkey
            );

            ALTER TABLE customer DROP COLUMN c_custkey;
            ALTER TABLE customer DROP COLUMN c_nationkey;
            ALTER TABLE customer RENAME COLUMN custkey_uuid TO c_custkey;
            ALTER TABLE customer RENAME COLUMN c_nationkey_uuid TO c_nationkey;

            ALTER TABLE orders DROP COLUMN o_orderkey;
            ALTER TABLE orders DROP COLUMN o_custkey;
            ALTER TABLE orders RENAME COLUMN orderkey_uuid TO o_orderkey;
            ALTER TABLE orders RENAME COLUMN o_custkey_uuid TO o_custkey;

            ALTER TABLE lineitem DROP COLUMN l_orderkey;
            ALTER TABLE lineitem DROP COLUMN l_partkey;
            ALTER TABLE lineitem DROP COLUMN l_suppkey;
            ALTER TABLE lineitem RENAME COLUMN l_orderkey_uuid TO l_orderkey;
            ALTER TABLE lineitem RENAME COLUMN l_partkey_uuid TO l_partkey;
            ALTER TABLE lineitem RENAME COLUMN l_suppkey_uuid TO l_suppkey;

            ALTER TABLE part DROP COLUMN p_partkey;
            ALTER TABLE part RENAME COLUMN partkey_uuid TO p_partkey;

            ALTER TABLE supplier DROP COLUMN s_suppkey;
            ALTER TABLE supplier DROP COLUMN s_nationkey;
            ALTER TABLE supplier RENAME COLUMN suppkey_uuid TO s_suppkey;
            ALTER TABLE supplier RENAME COLUMN s_nationkey_uuid TO s_nationkey;

            ALTER TABLE partsupp DROP COLUMN ps_partkey;
            ALTER TABLE partsupp DROP COLUMN ps_suppkey;
            ALTER TABLE partsupp RENAME COLUMN ps_partkey_uuid TO ps_partkey;
            ALTER TABLE partsupp RENAME COLUMN ps_suppkey_uuid TO ps_suppkey;

            ALTER TABLE nation DROP COLUMN n_nationkey;
            ALTER TABLE nation DROP COLUMN n_regionkey;
            ALTER TABLE nation RENAME COLUMN nationkey_uuid TO n_nationkey;
            ALTER TABLE nation RENAME COLUMN n_regionkey_uuid TO n_regionkey;

            ALTER TABLE region DROP COLUMN r_regionkey;
            ALTER TABLE region RENAME COLUMN regionkey_uuid TO r_regionkey;


        """
        con.sql(query_tpch)
        con.close()