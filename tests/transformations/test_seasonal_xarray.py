"""Tests for SeasonalXArray transformation."""
import pandas as pd
import pytest
import xarray

from chap_pymc.transformations.seasonal_xarray import SeasonalXArray, SeasonInformation


@pytest.fixture
def weekly_data() -> pd.DataFrame:
    """Create weekly time series data for testing.

    Returns DataFrame with columns: location, time_period, disease_cases
    """
    import numpy as np

    # Create 3 years of weekly data (156 weeks total)
    locations = ['LocationA', 'LocationB']
    years = [2020, 2021, 2022]
    weeks = list(range(1, 53))  # 52 weeks per year

    data = []
    for location in locations:
        for year in years:
            for week in weeks:
                time_period = f'{year}-{week:02d}'
                # Create seasonal pattern with 52-week period
                disease_cases = 10 + 5 * np.sin(2 * np.pi * week / 52) + np.random.randn() * 0.5
                data.append({
                    'location': location,
                    'time_period': time_period,
                    'disease_cases': max(0, disease_cases)
                })

    return pd.DataFrame(data)


# class TestSeasonalXArrayInit:
#     """Test initialization and parameter handling."""
#
#     def test_init_with_default_params(self, simple_monthly_data):
#         """Test initialization with default parameters."""
#         # TODO: Verify SeasonalXArray initializes correctly with defaults
#         # - Check that month and year columns are created
#         # - Verify data is copied (not referenced)
#         raise NotImplementedError("Test needs implementation")
#
#     def test_init_with_custom_params(self, simple_monthly_data):
#         """Test initialization with custom Params."""
#         # TODO: Test with custom Params (e.g., different alignment)
#         raise NotImplementedError("Test needs implementation")
#
#     def test_time_period_parsing(self, simple_monthly_data):
#         """Test that time_period strings are parsed correctly into month and year."""
#         # TODO: Verify '2020-01' becomes month=1, year=2020
#         raise NotImplementedError("Test needs implementation")


class Properties:
    """Test properties of SeasonalXArray."""

    def __init__(self, params: SeasonalXArray.Params) -> None:
        self._params = params
        self._season_info = SeasonInformation.get(params.frequency)

    def number_of_nonnull_elements(self, df: pd.DataFrame, data_set: xarray.Dataset) -> None:
        '''Assert that the number of non-null elements in the data_array matches the DataFrame.'''
        var = self._params.target_variable
        count_df = df[var].notnull().sum()
        count_da = int(data_set[var].count().values)
        assert count_df == count_da

    def shape(self, df: pd.DataFrame, data_set: xarray.Dataset) -> None:
        '''Assert that the shape of the data_array matches expected shape from DataFrame.'''
        n_locations = df['location'].nunique()
        shape = data_set[self._params.target_variable].shape
        assert shape[0] == n_locations
        assert shape[-1] == self._season_info.season_length

    def last_value(self, df: pd.DataFrame, data_set: xarray.Dataset) -> None:
        '''Assert that the last value in the data_array matches the DataFrame.'''
        var = self._params.target_variable
        last_month_idx = (self._season_info.season_length-self._params.split_season_index-1) % self._season_info.season_length
        for location, group in df.groupby('location'):
            last_df = group.sort_values('time_period')[var].iloc[-1]
            last_da = data_set[var].sel(location=location).isel(epi_year=-1).isel(epi_offset=last_month_idx).values
            assert last_df == last_da, f"Mismatch for location {location}: df={last_df}, da={last_da} last_month_idx={last_month_idx}, split_season_index={self._params.split_season_index}"

@pytest.mark.parametrize("split_season_index", range(12))
def test_seasonal_xarray_properties(simple_monthly_data: pd.DataFrame, split_season_index: int) -> None:
    """Test properties of SeasonalXArray transformation."""
    params = SeasonalXArray.Params(split_season_index=split_season_index)
    transformer = SeasonalXArray(params)
    data_array, mapping = transformer.get_dataset(simple_monthly_data)

    # Check number of non-null elements
    properties = Properties(params)
    properties.number_of_nonnull_elements(simple_monthly_data, data_array)
    properties.shape(simple_monthly_data, data_array)
    properties.last_value(simple_monthly_data, data_array)

#
# class TestFrequencyHandling:
#     """Test frequency-related properties and methods."""
#
#     def test_freq_name_monthly(self, simple_monthly_data):
#         """Test freq_name property returns 'month' for monthly data."""
#         # TODO: Assert freq_name == 'month' when frequency='M'
#         raise NotImplementedError("Test needs implementation")
#
#     def test_freq_name_weekly(self, weekly_data):
#         """Test freq_name property returns 'week' for weekly data."""
#         # TODO: Assert freq_name == 'week' when frequency='W'
#         raise NotImplementedError("Test needs implementation")
#
#     def test_season_length_monthly(self, simple_monthly_data):
#         """Test season_length returns 12 for monthly data."""
#         # TODO: Assert season_length == 12 for monthly frequency
#         raise NotImplementedError("Test needs implementation")
#
#     def test_season_length_weekly(self, weekly_data):
#         """Test season_length returns 52 for weekly data."""
#         # TODO: Assert season_length == 52 for weekly frequency
#         raise NotImplementedError("Test needs implementation")
#
#     def test_unsupported_frequency_raises_error(self, simple_monthly_data):
#         """Test that unsupported frequencies raise ValueError."""
#         # TODO: Test with frequency='D' (daily) should raise ValueError
#         raise NotImplementedError("Test needs implementation")
#
#
# class TestMinMonthFinding:
#     """Test the _find_min_month method."""
#
#     def test_find_min_month_identifies_lowest_mean(self, simple_monthly_data):
#         """Test that _find_min_month correctly identifies month with lowest mean."""
#         # TODO: Create data with known minimum month
#         # - Verify the method returns the expected month
#         raise NotImplementedError("Test needs implementation")
#
#     def test_find_min_month_with_min_alignment(self, simple_monthly_data):
#         """Test _find_min_month with alignment='min'."""
#         # TODO: Test that min alignment returns the month with minimum mean
#         raise NotImplementedError("Test needs implementation")
#
#     @pytest.mark.skip(reason="Mid alignment not yet implemented")
#     def test_find_min_month_with_mid_alignment(self, simple_monthly_data):
#         """Test _find_min_month with alignment='mid' (when implemented)."""
#         # TODO: Test mid alignment calculation when implemented
#         raise NotImplementedError("Test needs implementation")
#
#
# class TestXArrayConversion:
#     """Test the xarray() method that performs the main transformation."""
#
#     def test_xarray_returns_dataarray(self, simple_monthly_data):
#         """Test that xarray() returns an xarray.DataArray."""
#         # TODO: Verify return type is xarray.DataArray
#         raise NotImplementedError("Test needs implementation")
#
#     def test_xarray_has_correct_shape(self, simple_monthly_data):
#         """Test that output has expected dimensions."""
#         # TODO: Verify shape matches (n_locations * n_years, season_length)
#         # For monthly: should have 12 seasonal columns
#         raise NotImplementedError("Test needs implementation")
#
#     def test_xarray_has_location_and_year_index(self, simple_monthly_data):
#         """Test that output has location and year in index."""
#         # TODO: Verify 'location' and 'year' are in the result
#         raise NotImplementedError("Test needs implementation")
#
#     def test_xarray_column_names(self, simple_monthly_data):
#         """Test that seasonal columns are named correctly."""
#         # TODO: Verify columns named 'seasonal_0', 'seasonal_1', ..., 'seasonal_11'
#         raise NotImplementedError("Test needs implementation")
#
#     def test_seasonal_index_calculation(self, simple_monthly_data):
#         """Test that seasonal indices are calculated correctly relative to min_month."""
#         # TODO: If min_month=3 (March), verify:
#         # - March maps to seasonal_0
#         # - April maps to seasonal_1
#         # - February maps to seasonal_11
#         raise NotImplementedError("Test needs implementation")
#
#     def test_xarray_preserves_data_values(self, simple_monthly_data):
#         """Test that transformation preserves the actual disease_cases values."""
#         # TODO: Verify that values in output match input disease_cases
#         # - Check specific known values are in correct seasonal position
#         raise NotImplementedError("Test needs implementation")
#
#
# class TestEdgeCases:
#     """Test edge cases and error conditions."""
#
#     def test_empty_dataframe(self):
#         """Test behavior with empty DataFrame."""
#         # TODO: Test with empty DataFrame, should handle gracefully or raise error
#         raise NotImplementedError("Test needs implementation")
#
#     def test_single_location(self, simple_monthly_data):
#         """Test with only one location."""
#         # TODO: Verify works correctly with single location
#         raise NotImplementedError("Test needs implementation")
#
#     def test_single_year(self, simple_monthly_data):
#         """Test with only one year of data."""
#         # TODO: Verify works correctly with minimal time span
#         raise NotImplementedError("Test needs implementation")
#
#     def test_missing_months(self):
#         """Test with incomplete data (missing some months)."""
#         # TODO: Test behavior when some months are missing
#         # - Should pivot_table handle NaN appropriately?
#         raise NotImplementedError("Test needs implementation")
#
#     def test_multiple_years_same_location(self, simple_monthly_data):
#         """Test with multiple years for same location."""
#         # TODO: Verify each location-year combination gets separate row
#         raise NotImplementedError("Test needs implementation")
#
#
# class TestIntegration:
#     """Integration tests with realistic data."""
#
#     @pytest.mark.skip(reason="Requires real-world test data")
#     def test_with_thailand_data(self, thailand_ds):
#         """Test transformation with real Thailand dataset."""
#         # TODO: Use actual Thailand data from fixtures
#         # - Verify transformation produces expected structure
#         raise NotImplementedError("Test needs implementation")
#
#     @pytest.mark.skip(reason="Requires real-world test data")
#     def test_with_vietnam_data(self, data_path):
#         """Test transformation with real Vietnam dataset."""
#         # TODO: Use actual Vietnam data from fixtures
#         raise NotImplementedError("Test needs implementation")
#
#
# # Parametrized test example
# @pytest.mark.parametrize("frequency,expected_length", [
#     ("M", 12),
#     ("W", 52),
# ])
# def test_season_length_parametrized(frequency, expected_length, simple_monthly_data):
#     """Test season_length for different frequencies using parametrization."""
#     # TODO: Implement parametrized test for multiple frequencies
#     raise NotImplementedError("Test needs implementation")


class TestWeeklyData:
    """Test SeasonalXArray with weekly frequency data."""

    def test_weekly_season_length(self, weekly_data: pd.DataFrame) -> None:
        """Test that season_length is 52 for weekly data."""
        params = SeasonalXArray.Params(frequency='W')
        transformer = SeasonalXArray(params)
        assert transformer.season_length == 52

    def test_weekly_freq_name(self, weekly_data: pd.DataFrame) -> None:
        """Test that freq_name is 'week' for weekly data."""
        params = SeasonalXArray.Params(frequency='W')
        transformer = SeasonalXArray(params)
        assert transformer.freq_name == 'week'

    def test_weekly_transformation_shape(self, weekly_data: pd.DataFrame) -> None:
        """Test that weekly data transforms to correct shape."""
        params = SeasonalXArray.Params(frequency='W')
        transformer = SeasonalXArray(params)
        dataset, mapping = transformer.get_dataset(weekly_data)

        # Check shape: should have 52 weeks in the epi_offset dimension
        assert dataset['disease_cases'].shape[-1] == 52
        # Check that we have 2 locations
        assert dataset['disease_cases'].shape[0] == 2

    def test_weekly_properties(self, weekly_data: pd.DataFrame) -> None:
        """Test properties of weekly SeasonalXArray transformation."""
        params = SeasonalXArray.Params(frequency='W', split_season_index=0)
        transformer = SeasonalXArray(params)
        dataset, mapping = transformer.get_dataset(weekly_data)

        # Check number of non-null elements
        properties = Properties(params)
        properties.number_of_nonnull_elements(weekly_data, dataset)
        properties.shape(weekly_data, dataset)
        properties.last_value(weekly_data, dataset)

    @pytest.mark.parametrize("split_season_index", [0, 13, 26, 39])
    def test_weekly_with_different_splits(self, weekly_data: pd.DataFrame, split_season_index: int) -> None:
        """Test weekly data with different season start weeks."""
        params = SeasonalXArray.Params(frequency='W', split_season_index=split_season_index)
        transformer = SeasonalXArray(params)
        dataset, mapping = transformer.get_dataset(weekly_data)

        # Verify properties hold regardless of split
        properties = Properties(params)
        properties.number_of_nonnull_elements(weekly_data, dataset)
        properties.shape(weekly_data, dataset)
