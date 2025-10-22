import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray

from chap_pymc.seasonal_transform import SeasonalTransform, TransformParameters
import logging

logger = logging.getLogger(__name__)

# Constants
MONTHS_PER_YEAR = 12

@dataclasses.dataclass
class ModelInputBase:
    """Base class for model input."""
    X: xarray.DataArray
    y: xarray.DataArray
    last_month: int

    def n_months(self) -> int:
        return self.y.shape[-1]

    def n_years(self) -> int:
        return self.y.shape[1]

    def coords(self) -> dict[str, Any]:
        """Extract PyMC coordinate dict from xarray DataArrays."""
        return {
            'location': self.y.coords['location'].values,
            'year': self.y.coords['year'].values,
            'month': self.y.coords['month'].values,
            'feature': self.X.coords['feature'].values
        }

@dataclasses.dataclass
class FullModelInput(ModelInputBase):
    """Model input with xarray DataArrays containing all coordinate information."""
    seasonal_pattern: xarray.DataArray  # (location, month)
    seasonal_errors: xarray.DataArray  # (location, month)

@dataclasses.dataclass
class FourierModelInput(ModelInputBase):
    added_last_year: bool = False


class FourierInputCreator:
    """
    Creates xarray-based model input for SeasonalFourierRegression.

    Transforms raw training data into seasonal format with proper coordinates:
    1. Transforms data into seasonal format (locations, years, months)
    2. Extracts lagged temperature features
    3. Computes seasonal patterns
    4. Returns xarray DataArrays with full coordinate information
    """
    features = ['mean_temperature']

    def __init__(
        self,
        prediction_length: int = 3,
        lag: int = 3,
        mask_empty_seasons: bool = False
    ):
        """
        Initialize FourierInputCreator.

        Args:
            prediction_length: Number of months to predict ahead
            lag: Number of lagged temperature months to use as features
            mask_empty_seasons: Whether to mask seasons with low disease incidence
        """
        self._prediction_length = prediction_length
        self._lag = lag
        self._mask_empty_seasons = mask_empty_seasons
        self.seasonal_data: SeasonalTransform | None = None  # For backward compatibility

    def create_model_input(self, training_data: pd.DataFrame) -> FourierModelInput:
        """
        Transform raw training data into xarray-based model input.

        Args:
            training_data: DataFrame with columns ['location', 'time_period', 'disease_cases', 'mean_temperature', ...]

        Returns:
            ModelInput with xarray DataArrays containing coordinate information
        """
        training_data = training_data.copy()
        training_data['y'] = np.log1p(training_data['disease_cases']).interpolate()

        # Transform to seasonal format
        seasonal_data = SeasonalTransform(
            training_data,
            TransformParameters(
                min_prev_months=self._lag,
                min_post_months=self._prediction_length
            )
        )
        self.seasonal_data = seasonal_data  # Store for backward compatibility
        last_month = seasonal_data.last_seasonal_month_raw
        add_last_year = last_month +1+ self._prediction_length > MONTHS_PER_YEAR

        # Extract numpy arrays
        X = self.create_X(seasonal_data, add_last_year=add_last_year)
        y = seasonal_data.get_xarray('y', drop_first_year=True, add_last_year=add_last_year)
        if X.isnull().any():
            assert False,  f"NaNs found in feature array X: {X.where(X.isnull(), drop=True)}"



        logger.info(f"create_model_input: y.shape = {y.shape}")
        logger.info(f"create_model_input: last_seasonal_month = {last_month}")
        return FourierModelInput(
            X=X,
            y=y,
            last_month=last_month,
            added_last_year=add_last_year
        )

    def create_X(
        self,
        seasonal_data: SeasonalTransform,
        add_last_year: bool = False
    ) -> np.ndarray[tuple[Any, ...], np.dtype[np.float64]]:
        """
        Extract and standardize feature arrays (e.g., lagged temperature).

        Args:
            seasonal_data: SeasonalTransform object containing feature data

        Returns:
            Array of shape (locations, years, lag)
        """
        last_month = seasonal_data.last_seasonal_month_raw
        X = seasonal_data.get_xarray('mean_temperature', drop_first_year=not add_last_year, add_last_year=add_last_year)
        X = X.isel(month=slice(max(last_month-self._lag+1, 0), last_month + 1))
        X = X.rename({'month': 'feature'})
        std = X.std(dim=('year','feature'), skipna=True)
        mean = X.mean(dim=('year','feature'), skipna=True)
        X = (X - mean) / std
        if add_last_year:
            # Shift years up by one to drop the first incomplete year
            X_new = X.isel(year=slice(None, -1))
            X_new.coords['year'] = X.coords['year'][1:]
            X = X_new
        return X

def test_fourier_input_creator():
    """Test that FourierInputCreator produces xarray DataArrays with correct shapes and coordinates."""
    # Create synthetic data
    locations = ['LocationA', 'LocationB']
    n_months = 36  # 3 years of data
    dates = pd.date_range('2020-01', periods=n_months, freq='MS')

    data = []
    for loc in locations:
        for i, date in enumerate(dates):
            time_period = date.strftime('%Y-%m')
            disease_cases = np.sin(2 * np.pi * i / MONTHS_PER_YEAR) + 5 + np.random.randn() * 0.1
            mean_temperature = 20 + 10 * np.sin(2 * np.pi * i / MONTHS_PER_YEAR) + np.random.randn()
            data.append({
                'location': loc,
                'time_period': time_period,
                'disease_cases': max(0, disease_cases),
                'mean_temperature': mean_temperature
            })

    df = pd.DataFrame(data)

    # Create model input
    creator = FourierInputCreator(prediction_length=3, lag=3, mask_empty_seasons=False)
    model_input = creator.create_model_input(df)

    # Check that arrays are xarray DataArrays
    assert isinstance(model_input.X, xarray.DataArray)
    assert isinstance(model_input.y, xarray.DataArray)

    # Check shapes
    n_locations = len(locations)
    assert model_input.X.shape[0] == n_locations
    assert model_input.X.shape[2] == 3  # lag

    assert model_input.y.shape[0] == n_locations
    assert model_input.y.shape[2] == MONTHS_PER_YEAR  # months

    # Check dimensions
    assert model_input.X.dims == ('location', 'year', 'feature')
    assert model_input.y.dims == ('location', 'year', 'month')

    # Check coordinates exist
    assert 'location' in model_input.X.coords
    assert 'year' in model_input.X.coords
    assert 'feature' in model_input.X.coords
    assert 'month' in model_input.y.coords

    # Check coords() method
    coords = model_input.coords()
    assert set(coords.keys()) == {'location', 'year', 'month', 'feature'}

    print("✓ FourierInputCreator test passed")


# Backward compatibility alias
ModelInputCreator = FourierInputCreator
