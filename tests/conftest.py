import itertools
import logging
from pathlib import Path
from typing import Any, Generator

import chap_core
import pandas as pd
import pydantic
import pytest
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def configure_logging():
    """Configure logging for all tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        force=True  # Override any existing configuration
    )


@pytest.fixture
def df() -> pd.DataFrame:
    p = Path(__file__).parent / 'fixtures' / 'data' / 'training_data.csv'
    df = pd.read_csv(p)
    return df


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent / 'fixtures' / 'data'


@pytest.fixture
def large_df(data_path) -> pd.DataFrame:
    p = data_path / 'thailand.csv'
    return pd.read_csv(p)


@pytest.fixture
def thailand_ds(data_path) -> DataSet:
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
def viet_full_year(data_path) -> Generator[tuple[Any, Any], Any, None]:
    country = 'vietnam'
    return get_full_year(country, data_path)

@pytest.fixture
def nepal_full_year() -> Generator[tuple[Any, Any], Any, None]:
    data_path = Path('/Users/knutdr/Sources/chap_benchmarking/csv_datasets')
    return get_full_year('nepal_evaluation_set', data_path)

@pytest.fixture
def viet_first_instance(viet_full_year)->tuple[DataSet, DataSet]:
    return next(viet_full_year)

class Coords(pydantic.BaseModel):
    locations: list[str]
    years: list[int]
    variables: list[str]

@pytest.fixture
def simple_coords() -> Coords:
    locations = ['loc1', 'loc2']
    years = [2021, 2022]
    variables = ['disease_cases', 'mean_temperature']
    return Coords(locations=locations, years=years, variables=variables)

@pytest.fixture
def simple_monthly_data(simple_coords) -> pd.DataFrame:
    months = list(range(1, 13))
    rows = []
    for i, (location, year, month) in enumerate(itertools.product(simple_coords.locations, simple_coords.years, months)):
        rows.append({
            'location': location,
            'time_period': f'{year}-{month:02d}',
        } | {var: float(i*t + 1) for t, var in enumerate(simple_coords.variables)})
    return pd.DataFrame(rows)

@pytest.fixture
def weekly_data() -> pd.DataFrame:
    path = Path('/Users/knutdr/Sources/chap_benchmarking/csv_datasets')
    df = pd.read_csv(path / 'ewars_weekly.csv')
    return df

@pytest.fixture
def simple_future_data(simple_coords) -> pd.DataFrame:
    months = [1, 2, 3]
    rows = []
    for i, (location, month) in enumerate(itertools.product(simple_coords.locations, months)):
        rows.append({
            'location': location,
            'time_period': f'2023-{month:02d}',
        } | {var: float(i*t + 100) for t, var in enumerate(simple_coords.variables)})
    return pd.DataFrame(rows)

@pytest.fixture
def weekly_data() -> pd.DataFrame:
    """Create weekly time series data with date range format.

    Returns DataFrame with columns: location, time_period, disease_cases, mean_temperature
    Uses format: "YYYY-MM-DD/YYYY-MM-DD" for time_period
    """
    import numpy as np
    from datetime import datetime, timedelta

    # Create 3 years of weekly data
    locations = ['LocationA', 'LocationB']
    start_date = datetime(2020, 1, 6)  # First Monday of 2020
    weeks = 52 * 3  # 3 years

    data = []
    for location in locations:
        for week_num in range(weeks):
            week_start = start_date + timedelta(weeks=week_num)
            week_end = week_start + timedelta(days=6)

            # Create seasonal pattern with 52-week period
            week_of_year = week_start.isocalendar()[1]
            disease_cases = 10 + 5 * np.sin(2 * np.pi * week_of_year / 52) + np.random.randn() * 0.5
            mean_temperature = 20 + 10 * np.sin(2 * np.pi * week_of_year / 52) + np.random.randn()

            time_period = f'{week_start.strftime("%Y-%m-%d")}/{week_end.strftime("%Y-%m-%d")}'
            data.append({
                'location': location,
                'time_period': time_period,
                'disease_cases': max(0, disease_cases),
                'mean_temperature': mean_temperature
            })

    return pd.DataFrame(data)

def get_full_year(country, data_path: Path) -> Generator[tuple[Any, Any], Any, None]:
    csv_file = data_path / (f'{country}.csv')
    dataset = chap_core.data.DataSet.from_csv(csv_file)

    train_data, test_instances = train_test_generator(dataset, prediction_length=3, n_test_sets=12)
    i = 0
    for historic_data, _, future_data in test_instances:
        df = historic_data.to_pandas()
        f = future_data.to_pandas()
        f.time_period = f.time_period.astype(str)
        df.time_period = df.time_period.astype(str)
        logger.info(f'Yielding test instance {i}')
        i += 1
        yield df, f


def get_test_instance(country: str, data_path: Path, offset: int) -> pd.DataFrame:
    csv_file = data_path / (f'{country}.csv')
    dataset = chap_core.data.DataSet.from_csv(csv_file)

    train_data, _ = train_test_generator(dataset, prediction_length=3, n_test_sets=offset)
    df = train_data.to_pandas()
    df.time_period = df.time_period.astype(str)
    return df
