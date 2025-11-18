"""Tests for SeasonalFourierRegression"""
import pandas as pd
import pytest

from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
)
from chap_pymc.curve_parametrizations.fourier_parametrization_plots import plot_vietnam_faceted_predictions
from chap_pymc.inference_params import InferenceParams
from chap_pymc.models.seasonal_fourier_regression import (
    SeasonalFourierRegression,
    SeasonalFourierRegressionV2,
)
import logging

from chap_pymc.transformations.model_input_creator import FourierInputCreator

logger   = logging.getLogger(__name__)


def test_seasonal_fourier_regression_predict(viet_begin_season):
    """Test that SeasonalFourierRegression.predict works end-to-end"""
    # Create model with small MCMC params for fast testing
    model = SeasonalFourierRegression(
        prediction_length=3,
        lag=3,
        fourier_hyperparameters=FourierHyperparameters(n_harmonics=2),
        inference_params=InferenceParams(method='advi', n_iterations=10)
    )

    # Generate predictions
    predictions = model.predict(viet_begin_season, n_samples=100)

    # Check output structure
    assert isinstance(predictions, pd.DataFrame)
    assert 'location' in predictions.columns
    assert 'time_period' in predictions.columns
    assert 'sample_0' in predictions.columns

    # Check that we have predictions for all locations
    n_locations = viet_begin_season['location'].nunique()
    n_pred_months = 3
    assert len(predictions) == n_locations * n_pred_months

    # Check that predictions are positive (disease cases)
    sample_cols = [c for c in predictions.columns if c.startswith('sample_')]
    assert (predictions[sample_cols] >= 0).all().all()

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Locations: {predictions['location'].unique()}")
    print(f"Time periods: {predictions['time_period'].unique()}")
    print("\nSample predictions for first location:")
    print(predictions[predictions['location'] == predictions['location'].iloc[0]].head())

#@pytest.mark.slow
def test_viet_full_year(viet_full_year, full_inference_params, skip=7, country='vietnam', input_params=FourierInputCreator.Params()):
    for i, (viet_instance) in enumerate(viet_full_year):
        if i< skip:
            continue
        test_viet_regression(viet_instance, full_inference_params, i, country=country, input_params=input_params)

def test_nepal_full_year(nepal_full_year, full_inference_params, debug_inference_params, nepal_input_params):
    return test_viet_full_year(nepal_full_year, full_inference_params, skip=9, country='nepal', input_params=nepal_input_params)

@pytest.fixture
def full_inference_params():
    return InferenceParams(draws=1000,
                           tune=1000)

@pytest.fixture
def nepal_input_params():
    return FourierInputCreator.Params(skip_bottom_n_seasons=2)

@pytest.fixture
def debug_inference_params():
    return InferenceParams(draws=10, tune=10)

def test_viet_regression(viet_first_instance, full_inference_params, idx: int = 0, country='vietnam', input_params=FourierInputCreator.Params()):
    training_df, future_df = viet_first_instance
    logger.info(future_df['time_period'].min())
    model = SeasonalFourierRegressionV2(
        SeasonalFourierRegressionV2.Params(inference_params=full_inference_params, input_params=input_params),
        name=f'viet_regression_{idx}', )

    ds, mapping = model.get_input_data(future_df, training_df)
    samples = model.get_raw_samples(ds)
    assert not samples.isnull().any()
    median = samples.median(dim='samples')
    q_low = samples.quantile(0.1, dim='samples')
    q_high = samples.quantile(0.9, dim='samples')
    plot_vietnam_faceted_predictions(ds.y, median, q_low, q_high, ds.coords, output_file=f'{country}_regression_fit_{idx}.png')
    prediction_df = model.get_predictions_df(future_df, mapping, samples)
    assert not prediction_df['sample_1'].isnull().any(), prediction_df['sample_1'].unique()


def test_seasonal_fourier_regression_advi(viet_begin_season, truth=None):
    """Test that SeasonalFourierRegression.predict works with ADVI (faster variational inference)"""
    # Create model with ADVI for faster inference
    model = SeasonalFourierRegression(
        prediction_length=3,
        lag=3,
        fourier_hyperparameters=FourierHyperparameters(n_harmonics=2),
        inference_params=InferenceParams(method='advi', n_iterations=10)
    )

    # Generate predictions using ADVI
    predictions = model.predict(viet_begin_season, n_samples=100)

    # Check output structure
    assert isinstance(predictions, pd.DataFrame)
    assert 'location' in predictions.columns
    assert 'time_period' in predictions.columns
    n_locations = viet_begin_season['location'].nunique()
    n_pred_months = 3
    assert len(predictions) == n_locations * n_pred_months, f"Expected {n_locations * n_pred_months} predictions, got {len(predictions)}"

    # Check that predictions are positive
    sample_cols = [c for c in predictions.columns if c.startswith('sample_')]
    assert (predictions[sample_cols] >= 0).all().all()

    print(f"\nADVI Predictions shape: {predictions.shape}")


@pytest.mark.slow
def test_compare_with_seasonal_regression(viet_begin_season):
    """Compare SeasonalFourierRegression with original SeasonalRegression"""
    from chap_pymc.models.seasonal_regression import SeasonalRegression

    # Fit both models with same params
    inference_params = InferenceParams(method='hmc', chains=2, tune=100, draws=100)

    # Original model
    model_original = SeasonalRegression(
        prediction_length=3,
        lag=3,
        inference_params=inference_params
    )
    preds_original = model_original.predict(viet_begin_season)

    # Fourier model
    model_fourier = SeasonalFourierRegression(
        prediction_length=3,
        lag=3,
        fourier_hyperparameters=FourierHyperparameters(n_harmonics=2),
        inference_params=inference_params
    )
    preds_fourier = model_fourier.predict(viet_begin_season, n_samples=100)

    # Check both produce valid outputs
    assert preds_original.shape[0] == preds_fourier.shape[0]
    assert set(preds_original.columns[:2]) == set(preds_fourier.columns[:2])

    print("\nOriginal model predictions:")
    print(preds_original.head())
    print("\nFourier model predictions:")
    print(preds_fourier.head())


#@pytest.mark.skip(reason="SeasonalFourierRegressionV2 implementation incomplete")
def test_seasonal_fourier_regression_v2_basic(simple_monthly_data, simple_future_data):
    """Smoke test for SeasonalFourierRegressionV2 with simple data"""
    # Create V2 model with minimal params for fast testing
    model = SeasonalFourierRegressionV2(
        params=SeasonalFourierRegressionV2.Params(
            inference_params=InferenceParams(method='advi', n_iterations=10),
            fourier_hyperparameters=FourierHyperparameters(n_harmonics=1)
        )
    )

    # Call predict with training and future data
    result = model.predict(simple_monthly_data, simple_future_data)

    # Basic smoke test - verify it returns a DataFrame
    assert isinstance(result, pd.DataFrame)
    assert 'location' in result.columns
    assert 'time_period' in result.columns

    # Check that we have predictions for both locations
    assert result['location'].nunique() == 2

    print(f"\nV2 Predictions shape: {result.shape}")
    print(f"Columns: {result.columns.tolist()}")
