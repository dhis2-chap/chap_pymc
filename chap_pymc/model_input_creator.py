import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray

from chap_pymc.seasonal_transform import SeasonalTransform, TransformParameters

# Constants
MONTHS_PER_YEAR = 12


@dataclasses.dataclass
class ModelInput:
    X: np.ndarray
    y: np.ndarray
    seasonal_pattern: np.ndarray
    seasonal_errors: np.ndarray
    last_month: int

    def n_months(self) -> int:
        return self.y.shape[-1]


class ModelInputCreator:
    """
    Creates model input from raw training data by:
    1. Transforming data into seasonal format
    2. Extracting features (e.g., lagged temperature)
    3. Computing seasonal patterns
    """
    features = ['mean_temperature']

    def __init__(
        self,
        prediction_length: int = 3,
        lag: int = 3,
        mask_empty_seasons: bool = False
    ):
        self._prediction_length = prediction_length
        self._lag = lag
        self._mask_empty_seasons = mask_empty_seasons
        self.seasonal_data: SeasonalTransform | None = None

    def create_model_input(self, training_data: pd.DataFrame) -> ModelInput:
        """
        Transform raw training data into model input arrays.

        Args:
            training_data: DataFrame with columns ['location', 'time_period', 'disease_cases', 'mean_temperature', ...]

        Returns:
            ModelInput with arrays shaped (locations, years, months/features)
        """
        training_data = training_data.copy()
        training_data['y'] = np.log1p(training_data['disease_cases']).interpolate()


        seasonal_data = SeasonalTransform(
            training_data,
            TransformParameters(
                min_prev_months=self._lag,
                min_post_months=self._prediction_length
            )
        )
        self.seasonal_data = seasonal_data

        X = self.create_X(seasonal_data)
        y = seasonal_data['y']
        seasonal_pattern, std_per_month_per_loc = self.get_seasonal_pattern(y)
        last_month = seasonal_data.last_seasonal_month

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"create_model_input: y.shape = {y.shape}")
        logger.info(f"create_model_input: last_seasonal_month = {last_month}")
        logger.info(f"create_model_input: pad_left = {seasonal_data._pad_left}, pad_right = {seasonal_data._pad_right}")

        model_input = ModelInput(
            X=X,
            y=y,
            seasonal_pattern=seasonal_pattern,
            seasonal_errors=std_per_month_per_loc,
            last_month=last_month
        )
        return model_input

    def get_seasonal_pattern(
        self,
        y: np.ndarray[tuple[Any, ...], Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract seasonal pattern by standardizing each year and averaging across years.

        Args:
            y: Array of shape (locations, years, months)

        Returns:
            Tuple of (seasonal_pattern, std_per_month_per_loc), both shape (locations, 1, months)
        """
        loc_y = np.nanmean(y, axis=-1, keepdims=True)  # L, Y, 1
        scale_y = np.nanstd(y, axis=-1, keepdims=True)

        if self._mask_empty_seasons:
            mask = loc_y / loc_y.max(axis=1, keepdims=True) < 0.2
            loc_y[mask] = np.nan

        base = (y - loc_y) / np.maximum(scale_y, 0.001)  # L, Y, M

        # Compute standard deviation and mean pattern across years
        std_per_month_per_loc = np.nanstd(base, axis=1, keepdims=True)  # L, 1, M
        seasonal_pattern = np.nanmean(base, axis=1, keepdims=True)  # L, 1, M

        return seasonal_pattern, std_per_month_per_loc

    def _extract_lagged_features(
        self,
        feature_array: np.ndarray,
        last_month: int,
        lag: int
    ) -> np.ndarray:
        """
        Extract lagged features ending at last_month.

        Given a feature array with shape (locations, years, months) and a last_month index,
        extract the `lag` months ending at last_month (inclusive).

        Example:
            If last_month=5 and lag=3, extracts months [3, 4, 5]
            If last_month=2 and lag=3, extracts months [0, 1, 2]

        Args:
            feature_array: Array of shape (locations, years, months) containing feature values
            last_month: Index of the last month to include (0-indexed, after padding)
            lag: Number of lagged months to extract

        Returns:
            Array of shape (locations, years, lag) with lagged features
        """
        start_month = last_month - lag + 1
        end_month = last_month + 1  # +1 because slicing is exclusive at end
        lagged = feature_array[:, :, start_month:end_month]

        # Standardize features
        lagged = (lagged - np.nanmean(lagged)) / np.nanstd(lagged)
        return lagged

    def create_X(
        self,
        seasonal_data: SeasonalTransform
    ) -> np.ndarray[tuple[Any, ...], np.dtype[np.float64]]:
        """
        Extract and standardize feature arrays (e.g., lagged temperature).

        Args:
            seasonal_data: SeasonalTransform object containing feature data

        Returns:
            Array of shape (locations, years, lag)
        """
        last_month = seasonal_data.last_seasonal_month
        X = {feature: seasonal_data[feature] for feature in self.features}
        temp = self._extract_lagged_features(
            X['mean_temperature'],
            last_month,
            self._lag
        )
        return temp

    def to_xarray(self, model_input: ModelInput) -> ModelInput:
        """
        Convert ModelInput arrays to xarray DataArrays with named dimensions.
        This is useful for models that use pymc.dims for dimensional broadcasting.

        Args:
            model_input: ModelInput with numpy arrays

        Returns:
            ModelInput with xarray DataArrays
        """
        if self.seasonal_data is None:
            raise ValueError("Must call create_model_input first to populate seasonal_data")

        # Get coordinates from seasonal data and add feature names
        coords = self.seasonal_data.coords() | {
            'feature': [f'temp_lag{self._lag - i}' for i in range(self._lag)]
        }

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"to_xarray: model_input.y.shape = {model_input.y.shape}")
        logger.info(f"to_xarray: coords lengths - location={len(coords['location'])}, year={len(coords['year'])}, month={len(coords['month'])}")
        logger.info(f"to_xarray: coords['month'] = {coords['month']}")

        # Convert to xarray with named dimensions
        model_input.X = xarray.DataArray(
            model_input.X,
            dims=('location', 'year', 'feature'),
            coords={
                'location': coords['location'],
                'year': coords['year'],
                'feature': coords['feature']
            }
        )

        model_input.y = xarray.DataArray(
            model_input.y,
            dims=('location', 'year', 'month'),
            coords={
                'location': coords['location'],
                'year': coords['year'],
                'month': coords['month']
            }
        )

        logger.info(f"to_xarray: after creating xarray, y.shape = {model_input.y.shape}")

        # Squeeze middle dimension and convert to xarray
        model_input.seasonal_pattern = xarray.DataArray(
            model_input.seasonal_pattern[:, 0, :],
            dims=('location', 'month'),
            coords={
                'location': coords['location'],
                'month': coords['month']
            }
        )

        model_input.seasonal_errors = xarray.DataArray(
            model_input.seasonal_errors[:, 0, :],
            dims=('location', 'month'),
            coords={
                'location': coords['location'],
                'month': coords['month']
            }
        )

        return model_input


def test_model_input_creator():
    """Test that ModelInputCreator produces the expected output shapes."""
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
    creator = ModelInputCreator(prediction_length=3, lag=3, mask_empty_seasons=False)
    model_input = creator.create_model_input(df)

    # Check shapes
    n_locations = len(locations)
    assert model_input.X.ndim == 3
    assert model_input.X.shape[0] == n_locations
    assert model_input.X.shape[2] == 3  # lag

    assert model_input.y.ndim == 3
    assert model_input.y.shape[0] == n_locations
    assert model_input.y.shape[2] == MONTHS_PER_YEAR  # months

    assert model_input.seasonal_pattern.shape == (n_locations, 1, MONTHS_PER_YEAR)
    assert model_input.seasonal_errors.shape == (n_locations, 1, MONTHS_PER_YEAR)

    # Check that seasonal_data was stored
    assert creator.seasonal_data is not None
    print("✓ ModelInputCreator test passed")


def test_model_input_to_xarray():
    """Test that to_xarray conversion works correctly."""
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
    creator = ModelInputCreator(prediction_length=3, lag=3, mask_empty_seasons=False)
    model_input = creator.create_model_input(df)

    # Convert to xarray
    model_input_xr = creator.to_xarray(model_input)

    # Check that arrays are now xarray DataArrays
    assert isinstance(model_input_xr.X, xarray.DataArray)
    assert isinstance(model_input_xr.y, xarray.DataArray)
    assert isinstance(model_input_xr.seasonal_pattern, xarray.DataArray)
    assert isinstance(model_input_xr.seasonal_errors, xarray.DataArray)

    # Check dimensions
    assert model_input_xr.X.dims == ('location', 'year', 'feature')
    assert model_input_xr.y.dims == ('location', 'year', 'month')
    assert model_input_xr.seasonal_pattern.dims == ('location', 'month')
    assert model_input_xr.seasonal_errors.dims == ('location', 'month')

    # Check that seasonal_pattern and seasonal_errors were squeezed
    assert model_input_xr.seasonal_pattern.ndim == 2  # Was (L, 1, M), now (L, M)
    assert model_input_xr.seasonal_errors.ndim == 2

    # Check coordinates exist
    assert 'location' in model_input_xr.X.coords
    assert 'year' in model_input_xr.X.coords
    assert 'feature' in model_input_xr.X.coords
    assert 'month' in model_input_xr.y.coords

    print("✓ to_xarray test passed")
