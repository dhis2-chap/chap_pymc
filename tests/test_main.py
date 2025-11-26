"""Tests for main.py CLI functions."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from main import predict, detect_frequency


class TestDetectFrequency:
    """Test frequency detection from time_period format."""

    def test_detect_monthly(self):
        df = pd.DataFrame({'time_period': ['2024-01', '2024-02', '2024-03']})
        assert detect_frequency(df) == 'M'

    def test_detect_weekly_date_range(self):
        df = pd.DataFrame({'time_period': ['2024-01-01/2024-01-07', '2024-01-08/2024-01-14']})
        assert detect_frequency(df) == 'W'

    def test_detect_weekly_iso_format(self):
        df = pd.DataFrame({'time_period': ['2024-W01', '2024-W02', '2024-W03']})
        assert detect_frequency(df) == 'W'

    def test_detect_weekly_high_period(self):
        df = pd.DataFrame({'time_period': ['2024-15', '2024-16', '2024-17']})
        assert detect_frequency(df) == 'W'


@pytest.mark.slow
def test_predict_weekly_data():
    """Test predict function with weekly data files."""
    fixtures_path = Path(__file__).parent / 'fixtures'
    historic_data = str(fixtures_path / 'data' / 'weekly_trainig_data.py')
    future_data = str(fixtures_path / 'data' / 'weekly_future_data.py')
    config_file = str(fixtures_path / 'config' / 'debug_config.yaml')

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        out_file = f.name

    predict(
        model='fourier',
        historic_data=historic_data,
        future_data=future_data,
        out_file=out_file,
        model_config=config_file
    )

    # Verify output
    result = pd.read_csv(out_file)
    assert len(result) > 0
    assert 'location' in result.columns
    assert 'time_period' in result.columns

    # Check we have sample columns
    sample_cols = [c for c in result.columns if c.startswith('sample_')]
    assert len(sample_cols) > 0

    print(f"Predictions shape: {result.shape}")
    print(f"Locations: {result['location'].unique()}")

    # Cleanup
    Path(out_file).unlink()
