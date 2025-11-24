"""Minimal test skeleton for FourierInputCreator v2."""
import pytest
import xarray

from chap_pymc.transformations.model_input_creator import FourierInputCreator
from chap_pymc.transformations.seasonal_xarray import SeasonalXArray


class TestFourierInputCreatorV2:
    """Test the v2() method that uses SeasonalXArray."""

    def test_v2_basic_workflow(self, simple_monthly_data, simple_future_data):
        """Test basic v2 workflow with training and future data."""
        # Note: split_season_index must be None - v2() calculates it from future_data
        params = FourierInputCreator.Params(lag=3, seasonal_params=SeasonalXArray.Params())
        ds, mapping = FourierInputCreator(params=params).v2(simple_monthly_data, simple_future_data)
        assert isinstance(ds, xarray.Dataset)
        assert ds.X.shape[-1] == 3
        assert not ds.X.isnull().any()  # No NaNs in X
        # TODO: Test v2() method end-to-end
        # - Verify it returns FourierModelInput
        # - Check that future data NaNs are handled correctly
        # - Verify X and y shapes

    @pytest.mark.skip(reason="not implemented")
    def test_v2_handles_future_data_nans(self, simple_monthly_data, simple_future_data):
        """Test that v2 correctly handles NaN values in future data."""
        # TODO: Verify future_data['disease_cases'] is set to NaN
        # - Check dataset concatenation works
        raise NotImplementedError("Test needs implementation")

    @pytest.mark.skip(reason="not implemented")
    def test_v2_creates_correct_X_features(self, simple_monthly_data, simple_future_data):
        """Test that X features are extracted correctly with lag."""
        # TODO: Verify X contains lagged temperature features
        # - Check X has correct dimensions (location, season_idx, feature)
        # - Verify lag slicing: isel(season_idx=slice(-lag, None))
        raise NotImplementedError("Test needs implementation")

    @pytest.mark.skip(reason="not implemented")
    def test_v2_split_season_index(self, simple_monthly_data, simple_future_data):
        """Test that split_season_index is derived from future data."""
        # TODO: Verify split_season_index matches first month of future_data
        # - Parse time_period to get month
        # - Check SeasonalXArray uses correct split_season_index
        raise NotImplementedError("Test needs implementation")



@pytest.mark.skip(reason="not implemented")
class TestCreateModelInput:
    """Test the create_model_input() method (legacy interface)."""

    def test_create_model_input_basic(self, simple_monthly_data):
        """Test basic create_model_input workflow."""
        # TODO: Test legacy create_model_input() method
        # - Verify it returns FourierModelInput
        # - Check y is log-transformed: log1p(disease_cases)
        # - Verify X shape matches expectations
        raise NotImplementedError("Test needs implementation")

    def test_create_model_input_normalizes_y(self, simple_monthly_data):
        """Test that y is normalized (standardized)."""
        # TODO: Verify y_mean and y_std are computed
        # - Check y is standardized: (y - y_mean) / y_std
        # - Verify y_mean and y_std are stored in output
        raise NotImplementedError("Test needs implementation")

    def test_create_model_input_handles_last_year(self, simple_monthly_data):
        """Test that last year is added when needed."""
        # TODO: Test add_last_year logic
        # - When last_month + 1 + prediction_length > 12
        # - Verify added_last_year flag is set correctly
        raise NotImplementedError("Test needs implementation")

    def test_create_model_input_no_nans_in_X(self, simple_monthly_data):
        """Test that X contains no NaN values (raises AssertionError if found)."""
        # TODO: Verify assertion for NaNs in X
        # - Create data that would produce NaNs
        # - Verify AssertionError is raised
        raise NotImplementedError("Test needs implementation")

@pytest.mark.skip(reason="not implemented")
class TestFourierModelInputStructure:
    """Test FourierModelInput dataclass structure."""

    def test_fourier_model_input_has_required_fields(self):
        """Test that FourierModelInput has all required fields."""
        # TODO: Create a FourierModelInput instance
        # - Verify fields: X, y, last_month, added_last_year, prev_year_end, y_mean, y_std
        raise NotImplementedError("Test needs implementation")

    def test_n_months_returns_correct_value(self):
        """Test n_months() method."""
        # TODO: Create FourierModelInput with known y shape
        # - Verify n_months() returns y.shape[-1]
        raise NotImplementedError("Test needs implementation")

    def test_n_years_returns_correct_value(self):
        """Test n_years() method."""
        # TODO: Create FourierModelInput with known y shape
        # - Verify n_years() returns y.shape[1]
        raise NotImplementedError("Test needs implementation")

    def test_coords_returns_correct_dict(self):
        """Test coords() method returns proper coordinate dict."""
        # TODO: Create FourierModelInput with known coordinates
        # - Verify coords() returns dict with location, year, month, feature
        raise NotImplementedError("Test needs implementation")



@pytest.mark.skip(reason="not implemented")
class TestSeasonalDataProperty:
    """Test seasonal_data property."""

    def test_seasonal_data_raises_when_not_set(self):
        """Test that accessing seasonal_data before create_model_input raises error."""
        creator = FourierInputCreator()
        with pytest.raises(ValueError, match="seasonal_data has not been set yet"):
            _ = creator.seasonal_data

    def test_seasonal_data_available_after_create(self, simple_monthly_data):
        """Test that seasonal_data is accessible after create_model_input."""
        # TODO: Call create_model_input, then verify seasonal_data is available
        raise NotImplementedError("Test needs implementation")


# Integration test

@pytest.mark.skip(reason="not implemented")
def test_full_workflow_integration(simple_monthly_data, simple_future_data):
    """Integration test for complete workflow."""
    # TODO: Test full workflow from raw data to FourierModelInput
    # - Use simple_monthly_data and simple_future_data fixtures
    # - Call v2() method
    # - Verify all outputs are correctly structured
    # - Check X and y have compatible shapes
    raise NotImplementedError("Test needs implementation")
