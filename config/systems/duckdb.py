import os
import sys

from src.logger import get_logger
from src.models import System
from src.utils import get_tmp_path

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, root_directory)

import json
import os
from typing import Optional, Tuple

logger = get_logger(__name__)


def get_profile_path_duckdb(thread: int) -> str:
    return get_tmp_path(f'duckdb-profile-thread-{thread}.json')


def get_duckdb_profile_script(thread: int) -> str:
    path = get_profile_path_duckdb(thread)
    string = f"PRAGMA enable_profiling = 'json';pragma profile_output='{path}';"
    return string


def get_duckdb_runtime_and_cardinality(thread: int) -> Optional[Tuple[float, int]]:
    # load the json
    json_path = get_profile_path_duckdb(thread)

    # if the query crashed, the file does not exist
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r') as f:
        profile = json.load(f)

    # get the runtime, can be either in the timing or operator_timing field
    if 'timing' in profile:
        runtime = profile['timing']
        cardinality = profile['children'][0]['children'][0]['cardinality']
    elif 'operator_timing' in profile:
        runtime = profile['operator_timing']
        try:
            cardinality = profile['children'][0]['cardinality']
        except KeyError:  # Sometimes operator cardinality is not found.
            cardinality = profile['children'][0]['operator_cardinality']

    elif 'latency' in profile:
        runtime = profile['latency']
        cardinality = profile['result_set_size']

    else:
        # throw an error if the runtime is not found
        raise ValueError(f'Runtime not found in profile: {profile}')
    # delete the file
    os.remove(json_path)
    return runtime, cardinality


DUCK_DB_MAIN: System = {
    'version': 'duckdb_latest_main',
    'name': 'duckdb',
    'build_config': {
        'build_command': 'GEN=ninja BUILD_BENCHMARK=1 BUILD_HTTPFS=1 BUILD_TPCH=1 BUILD_PARQUET=1 BUILD_TPCDS=1 make',
        'location': {
            'location': 'github',
            # latest built as of July 8th
            'github_url': 'https://github.com/duckdb/duckdb/commit/223ff0a7dba7900039d5910a009247ef097fff3c',
        },
    },
    'run_config': {
        'run_file': 'build/release/duckdb <',
        'run_file_relative_to_build': True,
    },
    'setup_script': 'PRAGMA disable_progress_bar;',
    'set_threads_command': lambda n_threads: f"PRAGMA threads = {n_threads};",
    'get_start_profiler_command': get_duckdb_profile_script,
    'get_metrics': get_duckdb_runtime_and_cardinality,
}

DUCK_DB_FACT_INTERSECTION_METRICS: System = {
    **DUCK_DB_MAIN,
    'version': 'fact-intersection',
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/commit/446b25a1af4a39cede073f7f3872b49145ec2cd0',
        },
    },
}

DUCK_DB_FACT_INTERSECTION_NO_METRICS: System = {
    **DUCK_DB_MAIN,
    'version': 'fact-intersection-no-metrics',
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/commit/446b25a1af4a39cede073f7f3872b49145ec2cd0'
        },
    },
}

DUCK_DB_LP_JOIN_BASELINE: System = {
    **DUCK_DB_MAIN,
    'version': 'baseline',
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/commit/fd2e59672e02d49278e9491ed1bd8fa5d1cdb0a7'
        },
    },
}

DUCK_DB_LP_JOIN: System = {
    **DUCK_DB_MAIN,
    'version': 'lp-join',
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/commit/e30f8270594e6fde06ca87b2193ad31d91db047e'
        },
    },
}

DUCK_DB_LP_JOIN_NO_SALT: System = {
    **DUCK_DB_MAIN,
    'version': 'lp-join-no-salt',
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/commit/442274812bda9504b697524e58f796d182612838'
        },
    },
}

DUCK_DB_JOIN_OPTIMIZATION_BASELINE: System = {
    **DUCK_DB_MAIN,
    'version': 'join-optimization-baseline',
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/commit/4ba2e66277a7576f58318c1aac112faa67c47b11'
        },
    },
}

DUCK_DB_JOIN_OPTIMIZATION_HASH_MARKER_AND_COLLISION_BIT: System = {
    **DUCK_DB_MAIN,
    'version': 'join-optimization-hash-marker-and-collision-bit',
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/tree/join-optimization/hash-marker-and-collision-bit'
        },
    },
}

DUCK_DB_JOIN_OPTIMIZATION_HASH_MARKER: System = {
    **DUCK_DB_MAIN,
    'version': 'join-optimization-hash-marker',
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/tree/join-optimization/hash-marker'
        },
    },
}

DUCK_DB_NIGHTLY: System = {
    **DUCK_DB_MAIN,
    'version': 'nightly',
    'build_config': None,
    'run_config': {
        'run_file_relative_to_build': False,
        'run_file': '/Users/paul/.local/bin/duckman run nightly <',
    }
}

DUCK_DB_NIGHTLY_BUILD_LOCALLY: System = {
    **DUCK_DB_MAIN,
    'version': 'nightly-build-locally',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/duckdb/duckdb'
        },
    },
}

DUCK_DB_V100: System = {
    **DUCK_DB_MAIN,
    'version': 'v1.0.0',
    'build_config': None,
    'run_config': {
        'run_file_relative_to_build': False,
        'run_file': '/Users/paul/.local/bin/duckman run 1.0.0 <',
    }
}

DUCK_DB_V113: System = {
    **DUCK_DB_MAIN,
    'version': 'v1.1.3',
    'build_config': None,
    'run_config': {
        'run_file_relative_to_build': False,
        'run_file': '/Users/paul/.local/bin/duckman run 1.1.3 <',
    }
}

DUCK_DB_WITHOUT_ATOMICS: System = {
    **DUCK_DB_MAIN,
    'version': 'without-atomics',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/tree/join/atomics-test'
        },
    },
}

DUCK_DB_PARTITIONED: System = {
    **DUCK_DB_MAIN,
    'version': 'partitioned-ht',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/gropaul/duckdb/tree/join/partioning-non-atomic-v2'
        },
    },
}

DUCK_DB_USSR: System = {
    **DUCK_DB_MAIN,
    'version': 'USSR_Basic_implementation',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/ad1eaf9a8635a4d9297c5ad87396645387eb299b'
        },
    },
}

DUCK_DB_USSR_refactored: System = {
    **DUCK_DB_MAIN,
    'version': 'USSR_Basic_implementation_refactored',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/63abe1496d8c4f5bea975669b371cb6c8019b05e'
        },
    },
}

DUCK_DB_USSR_memcpy_optimization: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_memcpy_optimization',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/9e05da3147555e06ba4d77b60bfc87b1e9b649ba'
        },
    },
}

DUCK_DB_USSR_new_lock: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_new_lock',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/e81a7f50c4651279e417ada7e104f7b26378c99e'
        },
    },
}

DUCK_DB_USSR_predicate_materialize_once: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_predicate_materialize_once',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/d4e6c9bc332fb05cd01d8d0ea322453cc87e1ff3'
        },
    },
}

DUCK_DB_USSR_no_singleton: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_no_singleton',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/524c578ac1da5e4aebf422d2d66d67f911279320'
        },
    },

}
DUCK_DB_USSR_no_singleton_new_api: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_no_singleton_new_api',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/a0d71ba95e2bf39d9a7a776427c643ad7090df1b'
        },
    },
}

DUCK_DB_USSR_upper_limit: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_upper_limit',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/b7a404a2b06031eebbf236d1158c8dacbf659b11'
        },
    },
}

DUCK_DB_USSR_upper_limit_cheaper_hash: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_upper_limit_cheaper_hash',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/fab6b034f01d8e4734fd83ddbba3d34a580e079c'
        },
    },
}

DUCK_DB_USSR_stable_version: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_stable_version',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/6aa18b3f536c4134a52942a852412631c106d5ad'
        },
    },
}
DUCK_DB_USSR_stable_version_2x: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_stable_version_2x',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/db53180bc08af20df79f81b65a1d474cf6f4a55c'
        },
    },
}

DUCK_DB_USSR_stable_version_operator: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_stable_version_operator',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/412cfef1de7498ad21a13934bd61163d78550b2b'
        },
    },
}

DUCK_DB_USSR_stable_version_operator_bttr_strs_col_seg: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_stable_version_operator_bttr_strs_col_seg',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/88d92557fc57636c7db1fb20d8896791a1cac100'
        },
    },
}

DUCK_DB_USSR_stable_version_operator_bttr_strs_flattening: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_stable_version_operator_bttr_strs_flattening',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/0a5805dbe6115f4ccdf0fc1da287499b3aafc0b7'
        },
    },
}

DUCK_DB_USSR_stable_version_operator_bttr_strs_local: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCK_DB_USSR_stable_version_operator_bttr_strs_local',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/622273903fdd0c1113e92def80b7e6aac1936f5c'
        },
    },
}

DUCKDB_emit_DICT: System = {
    **DUCK_DB_MAIN,
    'version': 'DUCKDB_emit_DICT',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/8185e0309827930839f9fafb070bdb814c7fcb92'
        },
    },
}

ussr_2x_size: System = {
    **DUCK_DB_MAIN,
    'version': 'ussr_2x_size',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/68c47adcb1ec0193bbfb320f91903b8743e9afe2'
        },
    },
}

ussr_2x_size_analyze_all_vecs: System = {
    **DUCK_DB_MAIN,
    'version': 'ussr_2x_size_analyze_all_vecs',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/37ba6753d5ede94d4fb81bd866764e5cec94650b'
        },
    },
}

ussr_only_extra_branches: System = {
    **DUCK_DB_MAIN,
    'version': 'ussr_only_extra_branches',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/6804af683c78340bd47514755d02174c80eff943'
        },
    },
}

main_extra_computation: System = {
    **DUCK_DB_MAIN,
    'version': 'main_extra_computation',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/3a76094348fc5a6002ea22e21f2d076c9c6be526'
        },
    },
}
ussr_16MB: System = {
    **DUCK_DB_MAIN,
    'version': 'ussr_16MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/310071fe3f0605af6c34d9af3746220cecc4fb90'
        },
    },
}

UnifiedStringDictionary_1MB: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_1MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/6c898d0b3f3826378901748ebb4871bda2f46e88'
        },
    },
}

UnifiedStringDictionary_2MB: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_2MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/tree/UnifiedStringDictionary-2MB'
        },
    },
}

UnifiedStringDictionary_4MB: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_4MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/tree/UnifiedStringDictionary-4MB'
        },
    },
}

UnifiedStringDictionary_8MB: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_8MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/tree/UnifiedStringDictionary-8MB'
        },
    },
}
UnifiedStringDictionary_16MB: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_16MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/01bfcec016de40dc93fcf8233070aeabea54653a'
        },
    },
}

UnifiedStringDictionary_16MB_dev_with_clientcontext: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_16MB_dev_with_clientcontext',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/63d1105bf52d0a26f5996d00668338e856e13068'
        },
    },
}

UnifiedStringDictionary_lock_free_512K: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_lock_free_512K',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/2351b3ee52d2ad65e120cbc99bf59dd10fbde554'
        },
    },
}

UnifiedStringDictionary_initial_benchmark: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_initial_benchmark',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/1ce26eef6ddd50943985c3703925bd3e73e9fc82'
        },
    },
}

UnifiedStringDictionary_initial_benchmark_32MB: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_initial_benchmark_32MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/fdd377c37b7d1c084407632307526822a0a1b719'
        },
    },
}

UnifiedStringDictionary_initial_benchmark_64MB: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_initial_benchmark_64MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/572a37ba2bd406a40d28ddfefe21064a7f95bfd4'
        },
    },
}

UnifiedStringDictionary_initial_benchmark_32MB_upper_limit: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_initial_benchmark_32MB_upper_limit',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/00a0fc17a743e6b342c9037a650f2f302503fc5c'
        },
    },
}

UnifiedStringDictionary_initial_benchmark_32MB_upper_limit_smarter_insertion: System = {
    **DUCK_DB_MAIN,
    'version': 'unified_string_dictionary_sampling_32MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/c0db521d36d0a7d557231e69a4920bfb14c9edda'
        },
    },
}

UnifiedStringDictionary_1GB_full_insertion: System = {
    **DUCK_DB_MAIN,
    'version': 'UnifiedStringDictionary_1GB_full_insertion',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/727a64e4dcc882aa5b8fe1dc066f31e053eaaf18'
        },
    },
}

USSR_SALT_CLEAN: System = {
    **DUCK_DB_MAIN,
    'version': 'USSR_SALT_CLEAN',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/e0eb10075cda3183d7edd1a17c93dea2701a827d'
        },
    },
}


USSR_SALT_CLEAN_FLAT_VEC_JOIN: System = {
    **DUCK_DB_MAIN,
    'version': 'USSR_SALT_CLEAN_FLAT_VEC_JOIN',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/6e7db91eb0ed0b57ac2c1784ce74ea6f82591bc0'
        },
    },
}

USSR_SALT_CLEAN_FLAT_VEC_JOIN_NEW: System = {
    **DUCK_DB_MAIN,
    'version': 'USSR_SALT_CLEAN_FLAT_VEC_JOIN_NEW',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/56c20c556520756d07ae0de72f872f25b3e41e90'
        },
    },
}

Unified_String_Dictionary: System = {
    **DUCK_DB_MAIN,
    'version': 'unified_string_dictionary',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/c7804077a2c9e20e984482da58d0d0f378f52d32'
        },
    },
}


unified_string_dictionary_256MB_no_constraint: System = {
    **DUCK_DB_MAIN,
    'version': 'unified_string_dictionary_256MB',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/9941cbcc5596197e762ea0b49ceba36b5b94ab4b'
        },
    },
}

unified_string_dictionary_256MB_with_constraint: System = {
    **DUCK_DB_MAIN,
    'version': 'unified_string_dictionary_256MB_constraint',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/commit/11b2a2014a28e6746865815d9e00ab314af177d5'
        },
    },
}

unified_string_dictionary_sort_test: System = {
    **DUCK_DB_MAIN,
    'version': 'unified_string_dictionary_sort_test',
    'build_config': None,
    'build_config': {
        **DUCK_DB_MAIN['build_config'],
        'location': {
            'location': 'github',
            'github_url': 'https://github.com/OmidAfroozeh/duckdb/tree/refs/heads/USD_sort_test'
        },
    },
}
