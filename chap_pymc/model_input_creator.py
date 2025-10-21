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
    """Model input with xarray DataArrays containing all coordinate information."""
    X: xarray.DataArray  # (location, year, feature)
    y: xarray.DataArray  # (location, year, month)
    seasonal_pattern: xarray.DataArray  # (location, month)
    seasonal_errors: xarray.DataArray  # (location, month)
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

    def create_model_input(self, training_data: pd.DataFrame) -> ModelInput:
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

        # Extract numpy arrays
        X_np = self.create_X(seasonal_data)
        y_np = seasonal_data['y']
        seasonal_pattern_np, std_per_month_per_loc_np = self.get_seasonal_pattern(y_np)
        last_month = seasonal_data.last_seasonal_month

        # Get coordinate values
        locations = seasonal_data._df['location'].unique()
        n_years = y_np.shape[1]
        n_months = y_np.shape[2]

        coords_dict = {
            'location': locations,
            'year': np.arange(n_years),
            'month': np.arange(n_months),
            'feature': [f'temp_lag{self._lag - i}' for i in range(self._lag)]
        }

        # Convert to xarray DataArrays with proper coordinates
        X = xarray.DataArray(
            X_np,
            dims=('location', 'year', 'feature'),
            coords={
                'location': coords_dict['location'],
                'year': coords_dict['year'],
                'feature': coords_dict['feature']
            }
        )

        y = xarray.DataArray(
            y_np,
            dims=('location', 'year', 'month'),
            coords={
                'location': coords_dict['location'],
                'year': coords_dict['year'],
                'month': coords_dict['month']
            }
        )

        seasonal_pattern = xarray.DataArray(
            seasonal_pattern_np[:, 0, :],  # Squeeze middle dimension
            dims=('location', 'month'),
            coords={
                'location': coords_dict['location'],
                'month': coords_dict['month']
            }
        )

        seasonal_errors = xarray.DataArray(
            std_per_month_per_loc_np[:, 0, :],  # Squeeze middle dimension
            dims=('location', 'month'),
            coords={
                'location': coords_dict['location'],
                'month': coords_dict['month']
            }
        )

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"create_model_input: y.shape = {y.shape}")
        logger.info(f"create_model_input: last_seasonal_month = {last_month}")

        return ModelInput(
            X=X,
            y=y,
            seasonal_pattern=seasonal_pattern,
            seasonal_errors=seasonal_errors,
            last_month=last_month
        )

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
    assert isinstance(model_input.seasonal_pattern, xarray.DataArray)
    assert isinstance(model_input.seasonal_errors, xarray.DataArray)

    # Check shapes
    n_locations = len(locations)
    assert model_input.X.shape[0] == n_locations
    assert model_input.X.shape[2] == 3  # lag

    assert model_input.y.shape[0] == n_locations
    assert model_input.y.shape[2] == MONTHS_PER_YEAR  # months

    assert model_input.seasonal_pattern.shape == (n_locations, MONTHS_PER_YEAR)
    assert model_input.seasonal_errors.shape == (n_locations, MONTHS_PER_YEAR)

    # Check dimensions
    assert model_input.X.dims == ('location', 'year', 'feature')
    assert model_input.y.dims == ('location', 'year', 'month')
    assert model_input.seasonal_pattern.dims == ('location', 'month')
    assert model_input.seasonal_errors.dims == ('location', 'month')

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
