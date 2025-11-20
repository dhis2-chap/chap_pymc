import dataclasses
import logging
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import pytest
import xarray
from pydantic import Field
from xarray import Dataset

from chap_pymc.transformations.seasonal_transform import SeasonalTransform, TransformParameters
from chap_pymc.transformations.seasonal_xarray import SeasonalXArray, TimeCoords

logger = logging.getLogger(__name__)

# Constants
MONTHS_PER_YEAR = 12

@dataclasses.dataclass
class ModelInputBase:
    """Base class for model input."""
    X: xarray.DataArray
    y: xarray.DataArray
    last_month: int
    season_length: int = 12  # Default to months for backward compatibility

    def n_months(self) -> int:
        return self.y.shape[-1]

    def n_years(self) -> int:
        return self.y.shape[1]

    def coords(self) -> dict[str, Any]:
        """Extract PyMC coordinate dict from xarray DataArrays."""
        return {
            'location': self.y.coords['location'].values,
            'epi_year': self.y.coords['epi_year'].values,
            'epi_offset': self.y.coords['epi_offset'].values,
            'feature': self.X.coords['feature'].values
        }

@dataclasses.dataclass
class FullModelInput(ModelInputBase):
    """Model input with xarray DataArrays containing all coordinate information."""
    seasonal_pattern: xarray.DataArray | None = None  # (location, month)
    seasonal_errors: xarray.DataArray | None = None  # (location, month)

@dataclasses.dataclass
class FourierModelInput(ModelInputBase):
    added_last_year: bool = False
    prev_year_end: xarray.DataArray | None = None  # (location, year)
    y_mean: xarray.DataArray | None = None
    y_std: xarray.DataArray | None = None

@dataclasses.dataclass
class NormalizationParams:
    mean: xarray.DataArray
    std: xarray.DataArray

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
    class Params(pydantic.BaseModel):
        lag: int = 3
        seasonal_params: SeasonalXArray.Params = SeasonalXArray.Params()
        skip_bottom_n_seasons: int = 0

    def __init__(
        self,
        prediction_length: int = 3,
        lag: int = 3,
        params: Params = Params(),
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
        self._seasonal_data: SeasonalTransform | None = None  # For backward compatibility
        self._params = params
    @property
    def seasonal_data(self) -> SeasonalTransform:
        if self._seasonal_data is None:
            raise ValueError("seasonal_data has not been set yet.")
        return self._seasonal_data

    def v2(self, training_data: pd.DataFrame, future_data: pd.DataFrame) -> tuple[Dataset, tuple[dict[str, TimeCoords], NormalizationParams]]:
        """Backward compatibility method for previous interface."""
        future_data['disease_cases'] = np.nan
        data_frame = pd.concat([training_data, future_data], ignore_index=True)
        params = self._params.seasonal_params
        params.target_variable = 'y'
        assert params.split_season_index is None
        sx = SeasonalXArray(params)
        season_length = sx.season_length  # Extract season length from SeasonalXArray
        data_frame = data_frame.copy()
        data_frame['y'] = np.log1p(data_frame['disease_cases'])
        ds, mapping = sx.get_dataset(data_frame)
        y = ds['y']

        if self._params.skip_bottom_n_seasons > 0:
            # Calculate mean for each season across locations and offsets
            season_means = y.mean(dim=('location', 'epi_offset'))
            # Find the n seasons with lowest mean (argsort returns indices)
            args = season_means.argsort().values
            args = args[args!=0] #Don't remove first year
            bottom_season_indices = args[:self._params.skip_bottom_n_seasons]
            # Get the actual epi_year coordinate values at those indices
            bottom_season_coords = season_means.epi_year.values[bottom_season_indices]
            # Set those seasons to nan
            for season in bottom_season_coords:
                y.loc[dict(epi_year=season)] = np.nan


        y_mean = y.mean(dim=('epi_year', 'epi_offset'), skipna=True)
        y_std = y.std(dim=('epi_year', 'epi_offset'), skipna=True)
        # Set mean to 0 and std to 1 where they're NaN (no data or no variance)
        y_mean = y_mean.where(~y_mean.isnull(), 0.0)
        y_std = y_std.where(~y_std.isnull(), 1.0)
        n_params = NormalizationParams(
            mean=y_mean,
            std=y_std
        )
        assert not n_params.std.isnull().any(), n_params.std
        assert not n_params.mean.isnull().any(), n_params.mean
        y = (y-y_mean)/y_std

        # Calculate first period index based on data format
        first_period_str = str(future_data['time_period'].min())
        if '/' in first_period_str:
            # Weekly date range format: "2024-09-23/2024-09-29"
            start_date = pd.to_datetime(first_period_str.split('/')[0])
            day_of_year = start_date.timetuple().tm_yday
            first_month = (day_of_year - 1) // 7  # 0-indexed week
        else:
            # Monthly format: "2024-09"
            first_month = int(first_period_str.split('-')[1]) - 1
        #last_month = (first_month - 1) % 12
        last_month  = y.sizes['epi_offset']-y.isel(epi_year=-2).isnull().all(dim='location').values[::-1].argmin()-1
        # Get last_month value from each year for each location
        prev_year_y = y.sel(epi_offset=last_month)  # dims: (location, epi_year)
        # Roll values so epi_year=k contains the value from epi_year=k-1
        # shift(epi_year=1) moves values forward: index i gets value from index i-1
        # First year (epi_year=0) will be filled with NaN
        prev_year_y = prev_year_y.shift(epi_year=1)
        X = self.X_v2(training_data, first_month)

        # Subset y and prev_year_y to match the same epi_year as X
        y = y.sel(epi_year=X.epi_year)
        prev_year_y = prev_year_y.sel(epi_year=X.epi_year)

        assert X.shape[-1] == self._params.lag
        assert not X.isnull().any(), f"NaNs found in feature array X: {X.where(X.isnull(), drop=True)}"
        ds = xarray.Dataset({'X': X, 'y': y, 'prev_year_y': prev_year_y})
        ds.attrs['season_length'] = season_length  # Store season_length as attribute
        return (ds,
                (mapping,
                n_params))

    def X_v2(self, training_data: pd.DataFrame, first_month: int) -> xarray.DataArray:
        params = self._params.seasonal_params.copy()
        params.split_season_index = first_month
        ds_result = SeasonalXArray(params).get_dataset(training_data)[0]
        X = ds_result['mean_temperature']

        # Debug: check for NaNs after SeasonalXArray
        nan_count = int(X.isnull().sum().values)
        if nan_count > 0:
            print(f"X_v2 after SeasonalXArray: {nan_count} NaNs, shape={X.shape}")
            # Show which years have NaNs
            for year in X.epi_year.values:
                year_nans = int(X.sel(epi_year=year).isnull().sum().values)
                if year_nans > 0:
                    print(f"  epi_year {year}: {year_nans} NaNs")

        X = X.isel(epi_offset=slice(-self._lag, None))
        X = X.rename({'epi_offset': 'feature'})

        # Debug: check after slicing
        nan_count = int(X.isnull().sum().values)
        if nan_count > 0:
            print(f"X_v2 after slice to lag={self._lag}: {nan_count} NaNs, shape={X.shape}")

        X = (X - X.mean(dim=('epi_year','feature'))) / X.std(dim=('epi_year','feature'))

        # Debug: check after normalization
        nan_count = int(X.isnull().sum().values)
        if nan_count > 0:
            print(f"X_v2 after normalization: {nan_count} NaNs, shape={X.shape}")

        # Remove first year if missing predictors
        first_year_nans = int(X.isel(epi_year=0).isnull().sum().values)
        print(f"X_v2: first year (index 0, coord {X.epi_year.values[0]}) has {first_year_nans} NaNs")
        if X.isel(epi_year=0).isnull().any():
            print(f"X_v2: removing first year due to NaNs")
            X = X.isel(epi_year=slice(1, None))

        # Debug: final check
        nan_count = int(X.isnull().sum().values)
        if nan_count > 0:
            print(f"X_v2 final: {nan_count} NaNs, shape={X.shape}")

        return X

    def create_model_input(self, training_data: pd.DataFrame) -> FourierModelInput:
        """
        Transform raw training data into xarray-based model input.

        Args:
            training_data: DataFrame with columns ['location', 'time_period', 'disease_cases', 'mean_temperature', ...]

        Returns:
            ModelInput with xarray DataArrays containing coordinate information
        """
        training_data = training_data.copy()
        training_data['y'] = np.log1p(training_data['disease_cases'])

        # Transform to seasonal format
        seasonal_data = SeasonalTransform(
            training_data,
            TransformParameters(
                min_prev_months=self._lag,
                min_post_months=self._prediction_length
            )
        )
        self._seasonal_data = seasonal_data  # Store for backward compatibility
        last_month = seasonal_data.last_seasonal_month_raw
        add_last_year = last_month +1+ self._prediction_length > MONTHS_PER_YEAR

        # Extract numpy arrays
        X = self.create_X(seasonal_data, add_last_year=add_last_year)
        y = seasonal_data.get_xarray('y', drop_first_year=False, add_last_year=add_last_year)
        y_mean = y.mean(dim=('epi_year', 'epi_offset'))
        y_std = y.std(dim=('epi_year', 'epi_offset'))
        y = (y-y_mean)/y_std

        prev_year_end = y.isel(epi_offset=-1, epi_year=slice(None, -1))
        y = y.isel(epi_year=slice(1, None))  # Drop first year to align with X
        if X.isnull().any():
            raise AssertionError(f"NaNs found in feature array X: {X.where(X.isnull(), drop=True)}")



        logger.info(f"create_model_input: y.shape = {y.shape}")
        logger.info(f"create_model_input: last_seasonal_month = {last_month}")
        return FourierModelInput(
            X=X,
            y=y,
            last_month=last_month,
            added_last_year=add_last_year,
            prev_year_end=prev_year_end,
            y_mean=y_mean,
            y_std=y_std,
            season_length=MONTHS_PER_YEAR  # Using SeasonalTransform which is monthly-only
        )

    def create_X(
        self,
        seasonal_data: SeasonalTransform,
        add_last_year: bool = False
    ) -> xarray.DataArray:
        """
        Extract and standardize feature arrays (e.g., lagged temperature).

        Args:
            seasonal_data: SeasonalTransform object containing feature data

        Returns:
            Array of shape (locations, years, lag)
        """
        last_month = seasonal_data.last_seasonal_month_raw
        X = seasonal_data.get_xarray('mean_temperature', drop_first_year=not add_last_year, add_last_year=add_last_year)
        X = X.isel(epi_offset=slice(max(last_month-self._lag+1, 0), last_month + 1))
        X = X.rename({'epi_offset': 'feature'})
        std = X.std(dim=('epi_year','feature'), skipna=True)
        mean = X.mean(dim=('epi_year','feature'), skipna=True)
        X = (X - mean) / std
        if add_last_year:
            # Shift years up by one to drop the first incomplete year
            X_new = X.isel(epi_year=slice(None, -1))
            X_new.coords['epi_year'] = X.coords['epi_year'][1:]
            X = X_new
        return X


def test_fourier_input_creator(simple_df: tuple[pd.DataFrame, int]) -> None:
    """Test that FourierInputCreator produces xarray DataArrays with correct shapes and coordinates."""
    # Create synthetic data
    df, n_locations = simple_df

    # Create model input
    creator = FourierInputCreator(prediction_length=3, lag=3)
    model_input = creator.create_model_input(df)

    # Check that arrays are xarray DataArrays
    assert isinstance(model_input.X, xarray.DataArray)
    assert isinstance(model_input.y, xarray.DataArray)

    # Check shapes

    assert model_input.X.shape[0] == n_locations
    assert model_input.X.shape[2] == 3  # lag

    assert model_input.y.shape[0] == n_locations
    assert model_input.y.shape[2] == MONTHS_PER_YEAR  # months

    # Check dimensions
    assert model_input.X.dims == ('location', 'epi_year', 'feature')
    assert model_input.y.dims == ('location', 'epi_year', 'epi_offset')

    # Check coordinates exist
    assert 'location' in model_input.X.coords
    assert 'epi_year' in model_input.X.coords
    assert 'feature' in model_input.X.coords
    assert 'epi_offset' in model_input.y.coords

    # Check coords() method
    coords = model_input.coords()
    assert set(coords.keys()) == {'location', 'epi_year', 'epi_offset', 'feature'}

    print("✓ FourierInputCreator test passed")

@pytest.fixture
def simpled_df() -> tuple[pd.DataFrame, int]:
    locations = ['LocationA', 'LocationB']
    n_locations = len(locations)
    n_months = 36  # 3 years of data
    dates = pd.date_range('2020-01', periods=n_months, freq='MS')
    time_periods = [date.strftime('%Y-%m') for date in dates]
    data = []
    for loc in locations:
        for i, time_period in enumerate(time_periods):
            # time_period = date.strftime('%Y-%m')
            disease_cases = np.sin(2 * np.pi * i / MONTHS_PER_YEAR) + 5 + np.random.randn() * 0.1
            mean_temperature = 20 + 10 * np.sin(2 * np.pi * i / MONTHS_PER_YEAR) + np.random.randn()
            data.append({
                'location': loc,
                'time_period': time_period,
                'disease_cases': max(0, disease_cases),
                'mean_temperature': mean_temperature
            })

    df = pd.DataFrame(data)
    return df, n_locations


# Backward compatibility alias
ModelInputCreator = FourierInputCreator
