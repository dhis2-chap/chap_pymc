from pathlib import Path

import pandas as pd
import pytest
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import chap_core
from chap_core.assessment.dataset_splitting import train_test_generator

@pytest.fixture
def df() -> pd.DataFrame:
    p = Path(__file__).parent.parent/ 'test_data' / 'training_data.csv'
    return pd.read_csv(p)

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
def thai_begin_season(data_path) -> pd.DataFrame:
    country = 'thailand'
    offset = 12
    df = get_test_instance(country, data_path, offset)
    return df

@pytest.fixture
def viet_begin_season(data_path) -> pd.DataFrame:
    country = 'vietnam'
    offset = 10
    df = get_test_instance(country, data_path, offset)
    return df

def get_test_instance(country: str, data_path: Path, offset: int) -> pd.DataFrame:
    csv_file = data_path / ('%s.csv' % country)
    dataset = chap_core.data.DataSet.from_csv(csv_file)

    train_data, _ = train_test_generator(dataset, prediction_length=3, n_test_sets=offset)
    df = train_data.to_pandas()
    df.time_period = df.time_period.astype(str)
    return df
