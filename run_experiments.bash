#!/bin/bash

nohup poetry run python experiments/UnifiedStringDictionary/duckdb_ussr_tpch.py > all_output_tpch.log 2>&1
nohup poetry run python experiments/UnifiedStringDictionary/Aggregation/aggregate_variable_length.py > all_output_var_length_microbench.log 2>&1
nohup poetry run python experiments/UnifiedStringDictionary/Aggregation/aggregate_variable_groupsize_skewed.py > all_output_var_grpsize_skewed_microbench.log 2>&1
nohup poetry run python experiments/UnifiedStringDictionary/Join/duckdb_ussr_jointest.py > all_output_jointest.log 2>&1

