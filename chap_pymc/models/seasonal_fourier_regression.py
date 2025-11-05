"""
SeasonalFourierRegression - Fourier-based seasonal disease forecasting model
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm

from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
    FourierParametrization,
)
from chap_pymc.inference_params import InferenceParams
from chap_pymc.transformations.model_input_creator import FourierInputCreator

logging.basicConfig(level=logging.INFO)


def create_output(training_pdf: pd.DataFrame, posterior_samples: np.ndarray, n_samples: int = 1000) -> pd.DataFrame:
    """
    Convert posterior samples to output DataFrame format.

    Args:
        training_pdf: Training data DataFrame with location and time_period columns
        posterior_samples: Array of shape (n_locations, n_months, n_samples)
        n_samples: Number of samples to include in output

    Returns:
        DataFrame with columns: location, time_period, sample_0, sample_1, ..., sample_N
    """
    n_samples = min(n_samples, posterior_samples.shape[-1])
    horizon = posterior_samples.shape[-2]
    locations = training_pdf['location'].unique()
    last_time_idx = training_pdf['time_period'].max()
    year, month = map(int, last_time_idx.split('-'))

    # Calculate future time periods
    raw_months = np.arange(horizon) + month
    new_months = (raw_months % 12) + 1
    new_years = year + raw_months // 12
    new_time_periods = [f'{y:d}-{m:02d}' for y, m in zip(new_years, new_months)]

    colnames = ['location', 'time_period'] + [f'sample_{i}' for i in range(n_samples)]
    rows = []

    for l_id, location in enumerate(locations):
        for t_id, time_period in enumerate(new_time_periods):
            samples = posterior_samples[l_id, t_id, -n_samples:]
            new_row = [location, time_period] + samples.tolist()
            rows.append(new_row)

    return pd.DataFrame(rows, columns=colnames)


class SeasonalFourierRegression:
    """
    Fourier-based seasonal regression model for disease forecasting.

    Uses harmonic components (Fourier series) to model seasonal patterns,
    with optional temperature features affecting each harmonic.
    """

    def __init__(
        self,
        prediction_length: int = 3,
        lag: int = 3,
        inference_params: InferenceParams = InferenceParams(),
        fourier_hyperparameters: FourierHyperparameters = FourierHyperparameters(),
        mask_empty_seasons: bool = False
    ) -> None:
        """
        Initialize SeasonalFourierRegression.

        Args:
            prediction_length: Number of months to predict
            lag: Number of lagged temperature features
            n_harmonics: Number of Fourier harmonics (not including baseline)
            inference_params: Inference parameters (HMC or ADVI)
            mask_empty_seasons: Whether to mask seasons with low disease incidence
        """
        self._prediction_length = prediction_length
        self._lag = lag
        self._fourier_hyperparameters = fourier_hyperparameters
        self._n_harmonics = fourier_hyperparameters.n_harmonics
        self._inference_params = inference_params
        self._mask_empty_seasons = mask_empty_seasons
        self._seasonal_data = None

    def predict(
        self,
        training_data: pd.DataFrame,
        n_samples: int = 1000,
        return_inference_data: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, Any]:
        """
        Fit Fourier model and generate predictions for the next prediction_length months.

        Uses either HMC or ADVI based on inference_params.method.

        Args:
            training_data: DataFrame with columns: location, time_period, disease_cases, mean_temperature
            n_samples: Number of posterior samples to return (or samples to draw from ADVI approximation)
            return_inference_data: Whether to return the InferenceData object (or approximation for ADVI)

        Returns:
            DataFrame with predictions (and optionally InferenceData/approximation object)
        """
        # Create model input (returns xarray DataArrays directly)
        creator = FourierInputCreator(
            prediction_length=self._prediction_length,
            lag=self._lag,
        )
        model_input = creator.create_model_input(training_data)
        self.model_input = model_input

        # Set up model coordinates from model_input + harmonic dimension
        coords = model_input.coords() | {
            'harmonic': np.arange(0, self._n_harmonics + 1)  # Include baseline (h=0)
        }
        self.stored_coords = coords  # For potential inspection later
        logging.info("PyMC coords being passed:")
        for k, v in coords.items():
            logging.info(f"  {k}: len={len(v) if hasattr(v, '__len__') else 'N/A'}, values={v if len(v) < 20 else f'{list(v[:5])}...{list(v[-2:])}'}")

        # Build and fit Fourier model
        logging.info("Building Fourier parametrization model...")
        with pm.Model(coords=coords):
            X = model_input.X
            y = model_input.y

            fourier_model = FourierParametrization(
                self._fourier_hyperparameters
            )
            fourier_model.get_regression_model(X, y)

            # Choose inference method based on inference_params.method
            if self._inference_params.method == 'hmc':
                # HMC/NUTS sampling
                logging.info("Sampling from posterior using HMC/NUTS...")
                idata = pm.sample(**self._inference_params.model_dump(exclude={'method', 'n_iterations'}))

            else:  # 'advi'
                # ADVI variational inference
                logging.info("Fitting with ADVI...")
                # pm.model_to_graphviz(pm_model).render("fourier_model_graph", view=True)
                approx = pm.fit(n=self._inference_params.n_iterations, method='advi')

                # Sample from approximation
                logging.info("Sampling from approximation...")
                idata = approx.sample(n_samples)
            posterior_predictive = pm.sample_posterior_predictive(idata, var_names=['y_obs', 'A']).posterior_predictive

        # Extract predictions for unobserved months in last year
        logging.info("Extracting predictions...")
        # posterior = idata.posterior
        posterior = posterior_predictive
        predictions_xr = fourier_model.extract_predictions(posterior, model_input)
        predictions_xr = predictions_xr*model_input.y_std +model_input.y_mean
        # Select only the first prediction_length months
        predictions_xr = predictions_xr.isel(month=slice(0, self._prediction_length))

        # Flatten chains and draws: (chain, draw, location, month) -> (location, month, samples)
        predictions_samples = predictions_xr.stack(samples=('chain', 'draw')).values

        # Transform back from log space: y = exp(log(y+1)) - 1
        predictions_samples = np.expm1(predictions_samples)

        # Clamp negative values to 0 (disease cases can't be negative)
        predictions_samples = np.maximum(predictions_samples, 0)

        # Create output DataFrame
        predictions_df = create_output(training_data, predictions_samples, n_samples=n_samples)

        if return_inference_data:
            return predictions_df, idata
        else:
            return predictions_df
