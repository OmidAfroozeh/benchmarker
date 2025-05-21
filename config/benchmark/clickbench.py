import os
from typing import List

import duckdb

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path, pad

logger = get_logger(__name__)

CLICK_BENCH_QUERIES: List[dict] = [

    # {
    #     'name': 'q33',
    #     'index': 33,
    #     'run_script': {
    #         "duckdb": "SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    {
        'name': 'q34',
        'index': 34,
        'run_script': {
            "duckdb": "SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;"
        }
    },
    {
        'name': 'q34',
        'index': 34,
        'run_script': {
            "duckdb": "SELECT 1, URL, COUNT(*) AS c FROM hits_10p GROUP BY 1, URL ORDER BY c DESC LIMIT 10;"
        }
    },
    {
        'name': 'q34',
        'index': 34,
        'run_script': {
            "duckdb": "SELECT 1, URL, COUNT(*) AS c FROM hits_50p GROUP BY 1, URL ORDER BY c DESC LIMIT 10;"
        }
    },

    # {
    #     'name': 'q36',
    #     'index': 36,
    #     'run_script': {
    #         "duckdb": "SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q36',
    #     'index': 36,
    #     'run_script': {
    #         "duckdb": "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q37',
    #     'index': 37,
    #     'run_script': {
    #         "duckdb": "SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q38',
    #     'index': 38,
    #     'run_script': {
    #         "duckdb": "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;"
    #     }
    # },
    # {
    #     'name': 'q40',
    #     'index': 40,
    #     'run_script': {
    #         "duckdb": "SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;"
    #     }
    # },
    # {
    #     'name': 'q41',
    #     'index': 41,
    #     'run_script': {
    #         "duckdb": "SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;"
    #     }
    # },
    # {
    #     'name': 'q42',
    #     'index': 42,
    #     'run_script': {
    #         "duckdb": "SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000;"
    #     }
    # },
    # {
    #     'name': 'q43',
    #     'index': 43,
    #     'run_script': {
    #         "duckdb": "SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;"
    #     }
    # },
    # {
    #     'name': 'q44',
    #     'index': 44,
    #     'run_script': {
    #         "duckdb": "SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;"
    #     }
    # },
]

def get_clickbench() -> Benchmark:

    datasets: List[DataSet] = __generate_and_return_clickbenchdataset()


    queries = CLICK_BENCH_QUERIES

    return {
        'name': 'clickbench',
        'datasets': datasets,
        'queries': queries
    }


def __get_clickbench_file_path() -> str:
    file_name =  os.path.join('clickbench', f'clickbench-hits.db')
    return get_data_path(file_name)


def __generate_and_return_clickbenchdataset() -> List[DataSet]:
    __generate_clickbenchdataset()

    datasets: List[DataSet] = []

    duckdb_file_path = __get_clickbench_file_path()
    duckdb_file_name_without_extension = os.path.splitext(os.path.basename(duckdb_file_path))[0]
    setup_script = {
        'duckdb': f"ATTACH '{duckdb_file_path}' (READ_ONLY); USE '{duckdb_file_name_without_extension}';"
    }

    dataset: DataSet = {
        'name': f'clickbench-hits',
        'setup_script': setup_script,
        'config': {
        }
    }

    datasets.append(dataset)

    return datasets


def __generate_clickbenchdataset():

    logger.info(f'Downloading data for clickbench dataset')
    duckdb_file_path = __get_clickbench_file_path()

    # Only generate the data if the file does not exist
    if os.path.exists(duckdb_file_path):
        logger.info(f'File {duckdb_file_path} already exists, skipping...')
        return
    else:
        logger.info(f'File {duckdb_file_path} does not exist, generating...')




    con = duckdb.connect(duckdb_file_path)
    query_tpcds = f"""
        CREATE TABLE hits
(
    WatchID BIGINT NOT NULL,
    JavaEnable SMALLINT NOT NULL,
    Title TEXT,
    GoodEvent SMALLINT NOT NULL,
    EventTime TIMESTAMP NOT NULL,
    EventDate Date NOT NULL,
    CounterID INTEGER NOT NULL,
    ClientIP INTEGER NOT NULL,
    RegionID INTEGER NOT NULL,
    UserID BIGINT NOT NULL,
    CounterClass SMALLINT NOT NULL,
    OS SMALLINT NOT NULL,
    UserAgent SMALLINT NOT NULL,
    URL TEXT,
    Referer TEXT,
    IsRefresh SMALLINT NOT NULL,
    RefererCategoryID SMALLINT NOT NULL,
    RefererRegionID INTEGER NOT NULL,
    URLCategoryID SMALLINT NOT NULL,
    URLRegionID INTEGER NOT NULL,
    ResolutionWidth SMALLINT NOT NULL,
    ResolutionHeight SMALLINT NOT NULL,
    ResolutionDepth SMALLINT NOT NULL,
    FlashMajor SMALLINT NOT NULL,
    FlashMinor SMALLINT NOT NULL,
    FlashMinor2 TEXT,
    NetMajor SMALLINT NOT NULL,
    NetMinor SMALLINT NOT NULL,
    UserAgentMajor SMALLINT NOT NULL,
    UserAgentMinor VARCHAR(255) NOT NULL,
    CookieEnable SMALLINT NOT NULL,
    JavascriptEnable SMALLINT NOT NULL,
    IsMobile SMALLINT NOT NULL,
    MobilePhone SMALLINT NOT NULL,
    MobilePhoneModel TEXT,
    Params TEXT,
    IPNetworkID INTEGER NOT NULL,
    TraficSourceID SMALLINT NOT NULL,
    SearchEngineID SMALLINT NOT NULL,
    SearchPhrase TEXT,
    AdvEngineID SMALLINT NOT NULL,
    IsArtifical SMALLINT NOT NULL,
    WindowClientWidth SMALLINT NOT NULL,
    WindowClientHeight SMALLINT NOT NULL,
    ClientTimeZone SMALLINT NOT NULL,
    ClientEventTime TIMESTAMP NOT NULL,
    SilverlightVersion1 SMALLINT NOT NULL,
    SilverlightVersion2 SMALLINT NOT NULL,
    SilverlightVersion3 INTEGER NOT NULL,
    SilverlightVersion4 SMALLINT NOT NULL,
    PageCharset TEXT,
    CodeVersion INTEGER NOT NULL,
    IsLink SMALLINT NOT NULL,
    IsDownload SMALLINT NOT NULL,
    IsNotBounce SMALLINT NOT NULL,
    FUniqID BIGINT NOT NULL,
    OriginalURL TEXT,
    HID INTEGER NOT NULL,
    IsOldCounter SMALLINT NOT NULL,
    IsEvent SMALLINT NOT NULL,
    IsParameter SMALLINT NOT NULL,
    DontCountHits SMALLINT NOT NULL,
    WithHash SMALLINT NOT NULL,
    HitColor CHAR NOT NULL,
    LocalEventTime TIMESTAMP NOT NULL,
    Age SMALLINT NOT NULL,
    Sex SMALLINT NOT NULL,
    Income SMALLINT NOT NULL,
    Interests SMALLINT NOT NULL,
    Robotness SMALLINT NOT NULL,
    RemoteIP INTEGER NOT NULL,
    WindowName INTEGER NOT NULL,
    OpenerName INTEGER NOT NULL,
    HistoryLength SMALLINT NOT NULL,
    BrowserLanguage TEXT,
    BrowserCountry TEXT,
    SocialNetwork TEXT,
    SocialAction TEXT,
    HTTPError SMALLINT NOT NULL,
    SendTiming INTEGER NOT NULL,
    DNSTiming INTEGER NOT NULL,
    ConnectTiming INTEGER NOT NULL,
    ResponseStartTiming INTEGER NOT NULL,
    ResponseEndTiming INTEGER NOT NULL,
    FetchTiming INTEGER NOT NULL,
    SocialSourceNetworkID SMALLINT NOT NULL,
    SocialSourcePage TEXT,
    ParamPrice BIGINT NOT NULL,
    ParamOrderID TEXT,
    ParamCurrency TEXT,
    ParamCurrencyID SMALLINT NOT NULL,
    OpenstatServiceName TEXT,
    OpenstatCampaignID TEXT,
    OpenstatAdID TEXT,
    OpenstatSourceID TEXT,
    UTMSource TEXT,
    UTMMedium TEXT,
    UTMCampaign TEXT,
    UTMContent TEXT,
    UTMTerm TEXT,
    FromTag TEXT,
    HasGCLID SMALLINT NOT NULL,
    RefererHash BIGINT NOT NULL,
    URLHash BIGINT NOT NULL,
    CLID INTEGER NOT NULL
);
INSERT INTO hits BY NAME
SELECT *
    REPLACE (
        make_date(EventDate) AS EventDate,
        epoch_ms(EventTime * 1000) AS EventTime,
        epoch_ms(ClientEventTime * 1000) AS ClientEventTime,
        epoch_ms(LocalEventTime * 1000) AS LocalEventTime)
FROM read_parquet([format('https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{{}}.parquet', x) for x in range(0, 100)], binary_as_string=True);

       """

    con.sql(query_tpcds)
    con.close()

