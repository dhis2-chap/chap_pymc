from pathlib import Path

import pandas as pd
import pytest
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


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