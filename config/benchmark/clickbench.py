import os
from typing import List

import duckdb

from src.logger import get_logger
from src.models import DataSet, Benchmark, Query
from src.utils import get_data_path, pad

logger = get_logger(__name__)

CLICK_BENCH_QUERIES = [
    # {
    #     'name': 'q00',
    #     'index': 0,
    #     'run_script': {
    #         "duckdb": "SELECT COUNT(*) FROM hits;"
    #     }
    # },
    # {
    #     'name': 'q01',
    #     'index': 1,
    #     'run_script': {
    #         "duckdb": "SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;"
    #     }
    # },
    # {
    #     'name': 'q02',
    #     'index': 2,
    #     'run_script': {
    #         "duckdb": "SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;"
    #     }
    # },
    # {
    #     'name': 'q03',
    #     'index': 3,
    #     'run_script': {
    #         "duckdb": "SELECT AVG(UserID) FROM hits;"
    #     }
    # },
    # {
    #     'name': 'q04',
    #     'index': 4,
    #     'run_script': {
    #         "duckdb": "SELECT COUNT(DISTINCT UserID) FROM hits;"
    #     }
    # },
    # {
    #     'name': 'q05',
    #     'index': 5,
    #     'run_script': {
    #         "duckdb": "SELECT COUNT(DISTINCT SearchPhrase) FROM hits;"
    #     }
    # },
    # {
    #     'name': 'q06',
    #     'index': 6,
    #     'run_script': {
    #         "duckdb": "SELECT MIN(EventDate), MAX(EventDate) FROM hits;"
    #     }
    # },
    # {
    #     'name': 'q07',
    #     'index': 7,
    #     'run_script': {
    #         "duckdb": "SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;"
    #     }
    # },
    # {
    #     'name': 'q08',
    #     'index': 8,
    #     'run_script': {
    #         "duckdb": "SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q09',
    #     'index': 9,
    #     'run_script': {
    #         "duckdb": "SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT UserID) FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q10',
    #     'index': 10,
    #     'run_script': {
    #         "duckdb": "SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q11',
    #     'index': 11,
    #     'run_script': {
    #         "duckdb": "SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q12',
    #     'index': 12,
    #     'run_script': {
    #         "duckdb": "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q13',
    #     'index': 13,
    #     'run_script': {
    #         "duckdb": "SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q14',
    #     'index': 14,
    #     'run_script': {
    #         "duckdb": "SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q15',
    #     'index': 15,
    #     'run_script': {
    #         "duckdb": "SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q16',
    #     'index': 16,
    #     'run_script': {
    #         "duckdb": "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q17',
    #     'index': 17,
    #     'run_script': {
    #         "duckdb": "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q18',
    #     'index': 18,
    #     'run_script': {
    #         "duckdb": "SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q19',
    #     'index': 19,
    #     'run_script': {
    #         "duckdb": "SELECT UserID FROM hits WHERE UserID = 435090932899640449;"
    #     }
    # },
    # {
    #     'name': 'q20',
    #     'index': 20,
    #     'run_script': {
    #         "duckdb": "SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%';"
    #     }
    # },
    # {
    #     'name': 'q21',
    #     'index': 21,
    #     'run_script': {
    #         "duckdb": "SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q22',
    #     'index': 22,
    #     'run_script': {
    #         "duckdb": "SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q23',
    #     'index': 23,
    #     'run_script': {
    #         "duckdb": "SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q24',
    #     'index': 24,
    #     'run_script': {
    #         "duckdb": "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q25',
    #     'index': 25,
    #     'run_script': {
    #         "duckdb": "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q26',
    #     'index': 26,
    #     'run_script': {
    #         "duckdb": "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q27',
    #     'index': 27,
    #     'run_script': {
    #         "duckdb": "SELECT CounterID, AVG(STRLEN(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;"
    #     }
    # },
    # {
    #     'name': 'q28',
    #     'index': 28,
    #     'run_script': {
    #         "duckdb": "SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k, AVG(STRLEN(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;"
    #     }
    # },
    # {
    #     'name': 'q29',
    #     'index': 29,
    #     'run_script': {
    #         "duckdb": "SELECT SUM(ResolutionWidth)" + "".join([f", SUM(ResolutionWidth + {i})" for i in range(1, 90)]) + " FROM hits;"
    #     }
    # },
    # {
    #     'name': 'q30',
    #     'index': 30,
    #     'run_script': {
    #         "duckdb": "SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q31',
    #     'index': 31,
    #     'run_script': {
    #         "duckdb": "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    # {
    #     'name': 'q32',
    #     'index': 32,
    #     'run_script': {
    #         "duckdb": "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    {
        'name': 'q33',
        'index': 33,
        'run_script': {
            "duckdb": "SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10;"
        }
    },
    {
        'name': 'q34',
        'index': 34,
        'run_script': {
            "duckdb": "SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;"
        }
    },
    # {
    #     'name': 'q35',
    #     'index': 35,
    #     'run_script': {
    #         "duckdb": "SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10;"
    #     }
    # },
    {
        'name': 'q36',
        'index': 36,
        'run_script': {
            "duckdb": "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10;"
        }
    },
    {
        'name': 'q37',
        'index': 37,
        'run_script': {
            "duckdb": "SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10;"
        }
    },
    {
        'name': 'q38',
        'index': 38,
        'run_script': {
            "duckdb": "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;"
        }
    },
    # {
    #     'name': 'q39',
    #     'index': 39,
    #     'run_script': {
    #         "duckdb": "SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;"
    #     }
    # },
    # {
    #     'name': 'q40',
    #     'index': 40,
    #     'run_script': {
    #         "duckdb": "SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;"
    #     }
    # },
    # {
    #     'name': 'q41',
    #     'index': 41,
    #     'run_script': {
    #         "duckdb": "SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000;"
    #     }
    # },
    # {
    #     'name': 'q42',
    #     'index': 42,
    #     'run_script': {
    #         "duckdb": "SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;"
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

