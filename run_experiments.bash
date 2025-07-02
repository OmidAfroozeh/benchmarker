#!/bin/bash
poetry lock
poetry install

poetry run python experiments/UnifiedStringDictionary/duckdb_ussr_tpch.py
poetry run python experiments/UnifiedStringDictionary/Aggregation/aggregate_variable_length.py
poetry run python experiments/UnifiedStringDictionary/Aggregation/aggregate_variable_groupsize_skewed.py
poetry run python experiments/UnifiedStringDictionary/Join/duckdb_ussr_jointest.py
