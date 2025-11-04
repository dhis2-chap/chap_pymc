"""Tests for SeasonalFourierRegression"""
import pandas as pd

from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
)
from chap_pymc.inference_params import InferenceParams
from chap_pymc.models.seasonal_fourier_regression import SeasonalFourierRegression


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

def test_viet_full_year(viet_full_year):
    for i, (viet_instance, t) in enumerate(viet_full_year):
        if i<7:
            continue
        test_seasonal_fourier_regression_advi(viet_instance, t)


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
