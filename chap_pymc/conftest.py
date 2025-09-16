from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def df() -> pd.DataFrame:
    p = Path(__file__).parent.parent/ 'test_data' / 'training_data.csv'
    return pd.read_csv(p)
