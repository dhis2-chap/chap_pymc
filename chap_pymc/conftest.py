from pathlib import Path
import logging
import pandas as pd
import pytest
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import chap_core
from chap_core.assessment.dataset_splitting import train_test_generator
logger = logging.getLogger(__name__)
@pytest.fixture
def df() -> pd.DataFrame:
    p = Path(__file__).parent.parent/ 'test_data' / 'training_data.csv'
    df = pd.read_csv(p)
    return df

@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent.parent / 'test_data'

@pytest.fixture
def large_df(data_path) -> pd.DataFrame:
    p = data_path / 'thailand.csv'
    return pd.read_csv(p)

@pytest.fixture
def thailand_ds(data_path) -> pd.DataFrame:
    return DataSet.from_csv(data_path / 'thailand.csv')

@pytest.fixture
def colombia_df(data_path) -> pd.DataFrame:
    return pd.read_csv(data_path / 'colombia.csv')


@pytest.fixture
def thai_begin_season(data_path) -> pd.DataFrame:
    country = 'thailand'
    offset = 12
    df = get_test_instance(country, data_path, offset)
    return df

@pytest.fixture
def nepal_data() -> pd.DataFrame:
    country = 'nepal_evaluation_set'
    offset = 6
    local_data_path = Path('/Users/knutdr/Sources/chap_benchmarking/csv_datasets')
    df = get_test_instance(country, local_data_path, offset)
    return df

@pytest.fixture
def viet_begin_season(data_path) -> pd.DataFrame:
    country = 'vietnam'
    offset = 10
    df = get_test_instance(country, data_path, offset)
    return df

@pytest.fixture
def viet_full_year(data_path) -> pd.DataFrame:
    country = 'vietnam'
    return (df for df in get_full_year(country, data_path))

def get_full_year(country, data_path: Path) -> pd.DataFrame:
    csv_file = data_path / ('%s.csv' % country)
    dataset = chap_core.data.DataSet.from_csv(csv_file)

    train_data, test_instances = train_test_generator(dataset, prediction_length=3, n_test_sets=12)
    i = 0
    for historic_data, _, future_data  in test_instances:
        df = historic_data.to_pandas()
        f = future_data.to_pandas()
        f.time_period = f.time_period.astype(str)
        df.time_period = df.time_period.astype(str)
        logger.info('Yielding test instance %d' % i)
        i += 1
        yield df, f

def get_test_instance(country: str, data_path: Path, offset: int) -> pd.DataFrame:
    csv_file = data_path / ('%s.csv' % country)
    dataset = chap_core.data.DataSet.from_csv(csv_file)

    train_data, _ = train_test_generator(dataset, prediction_length=3, n_test_sets=offset)
    df = train_data.to_pandas()
    df.time_period = df.time_period.astype(str)
    return df
