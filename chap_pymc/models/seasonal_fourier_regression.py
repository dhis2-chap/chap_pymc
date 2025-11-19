"""
SeasonalFourierRegression - Fourier-based seasonal disease forecasting model
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import pymc as pm
import xarray
from pandas import DataFrame
from xarray import DataArray, Dataset

from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
    FourierParametrization,
)
from chap_pymc.curve_parametrizations.fourier_parametrization_plots import (
    plot_vietnam_faceted_predictions,
)
from chap_pymc.inference_params import InferenceParams
from chap_pymc.transformations.model_input_creator import (
    FourierInputCreator,
    NormalizationParams,
)
from chap_pymc.transformations.seasonal_xarray import TimeCoords
from chap_pymc.util import TARGET_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_output(training_pdf: pd.DataFrame, posterior_samples: np.ndarray, n_samples: int = 1000, season_length: int = 12) -> pd.DataFrame:
    """
    Convert posterior samples to output DataFrame format.

    Args:
        training_pdf: Training data DataFrame with location and time_period columns
        posterior_samples: Array of shape (n_locations, n_periods, n_samples)
        n_samples: Number of samples to include in output
        season_length: Number of periods per year (12 for months, 52 for weeks)

    Returns:
        DataFrame with columns: location, time_period, sample_0, sample_1, ..., sample_N
    """
    n_samples = min(n_samples, posterior_samples.shape[-1])
    horizon = posterior_samples.shape[-2]
    locations = training_pdf['location'].unique()
    last_time_idx = training_pdf['time_period'].max()
    year, period = map(int, last_time_idx.split('-'))

    # Calculate future time periods
    raw_periods = np.arange(horizon) + period
    new_periods = (raw_periods % season_length) + 1
    new_years = year + raw_periods // season_length
    new_time_periods = [f'{y:d}-{p:02d}' for y, p in zip(new_years, new_periods)]

    colnames = ['location', 'time_period'] + [f'sample_{i}' for i in range(n_samples)]
    rows = []

    for l_id, location in enumerate(locations):
        for t_id, time_period in enumerate(new_time_periods):
            samples = posterior_samples[l_id, t_id, -n_samples:]
            new_row = [location, time_period] + samples.tolist()
            rows.append(new_row)

    return pd.DataFrame(rows, columns=colnames)

class SeasonalFourierRegressionV2:
    class Params(pydantic.BaseModel):
        inference_params: InferenceParams = InferenceParams()
        fourier_hyperparameters: FourierHyperparameters = FourierHyperparameters()
        input_params: FourierInputCreator.Params = FourierInputCreator.Params()

    def __init__(self, params: Params = Params(), name: str|None=None) -> None:
        self._params = params
        self._name = name

    def predict(self, training_data: pd.DataFrame, future_data: pd.DataFrame,
                save_plot: bool = True, country: str = 'model') -> pd.DataFrame:
        ds, mapping = self.get_input_data(future_data, training_data)
        samples = self.get_raw_samples(ds)

        # Automatically plot predictions if requested and model has a name
        if save_plot and self._name is not None:
            first_future_period = str(future_data['time_period'].min())
            median = samples.median(dim='samples')
            q_low = samples.quantile(0.1, dim='samples')
            q_high = samples.quantile(0.9, dim='samples')
            output_file = TARGET_DIR / f'{country}_regression_fit_{first_future_period}.png'
            logger.info(output_file)
            plot_vietnam_faceted_predictions(ds.y, median, q_low, q_high, ds.coords, output_file=output_file)
        else:
            raise Exception()

        prediction_df = self.get_predictions_df(future_data, mapping, samples)
        return prediction_df

    def get_input_data(self, future_data: DataFrame, training_data: DataFrame) -> tuple[Dataset, tuple[dict[str, TimeCoords], NormalizationParams]]:
        ds, mappings = FourierInputCreator(params=self._params.input_params).v2(training_data, future_data)

        return ds, mappings

    def get_predictions_df(self, future_data: DataFrame, mappings: tuple[dict[str, TimeCoords], NormalizationParams], samples: DataArray) -> DataFrame:
        mapping, n_params = mappings
        samples_xr = samples*n_params.std+n_params.mean
        n_samples = self._params.inference_params.n_samples
        indices = np.random.choice(samples_xr.sizes['samples'], replace=True, size=n_samples)
        samples_np: np.ndarray[Any, Any] = np.maximum(0, np.expm1(samples_xr.isel(samples=indices)))
        colnames = ['location', 'time_period'] + [f'sample_{i}' for i in range(n_samples)]
        rows = []
        # Convert back to xarray for selection
        samples_final = xarray.DataArray(samples_np, coords=samples_xr.isel(samples=indices).coords, dims=samples_xr.dims)
        for row in future_data.itertuples():
            location = row.location
            time_period = row.time_period
            array_coords = mapping[str(time_period)]
            sample_values = samples_final.sel(location=location, **array_coords.model_dump()).values
            new_row = [location, time_period] + sample_values.tolist()
            rows.append(new_row)
        prediction_df = pd.DataFrame(rows, columns=colnames)
        return prediction_df

    def get_raw_samples(self, ds: xarray.Dataset) -> xarray.DataArray:
        season_length = ds.attrs.get('season_length', 12)  # Get season_length from Dataset attributes
        fourier_model = FourierParametrization(self._params.fourier_hyperparameters, season_length=season_length)
        # ds = ds.expand_dims(fourier_model.extra_dims)
        coords = {dim: ds[dim].values for dim in ds.dims} | fourier_model.extra_dims
        with pm.Model(coords=coords) as model:
            prev_year_y = ds.get('prev_year_y', None)  # Get from Dataset if available
            fourier_model.get_regression_model(ds.X, ds.y, prev_year_y=prev_year_y)

            # Choose inference method based on inference_params.method
            inference_params = self._params.inference_params
            if inference_params.method == 'hmc':
                idata = pm.sample(**inference_params.model_dump(exclude={'method', 'n_iterations'}))
            else:  # 'advi'
                approx = pm.fit(n=inference_params.n_iterations, method='advi')
                idata = approx.sample(inference_params.n_samples)
            posterior_predictive = pm.sample_posterior_predictive(idata, var_names=['y_obs', 'A']).posterior_predictive
        if self._name is not None:
            posterior_predictive.to_netcdf(TARGET_DIR / (self._name + '_posterior.nc'))
            idata.to_netcdf(TARGET_DIR / (self._name + 'idata.nc'))
            ds.to_netcdf(TARGET_DIR / (self._name + '_ds.nc'))
        # Extract predictions
        #arviz.plot_posterior(idata, var_names=['sigma', 'a_sigma'])
        #plt.show()
        #arviz.plot_posterior(idata, var_names=['a_mu'])
        #plt.show()
        samples: xarray.DataArray = posterior_predictive['y_obs'].stack(samples=('chain', 'draw'))
        return samples


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
        return_inference_data: bool = False,
        future_data: pd.DataFrame | None = None,
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
                self._fourier_hyperparameters,
                season_length=model_input.season_length
            )
            fourier_model.get_regression_model(X, y)

            # Choose inference method based on inference_params.method
            if self._inference_params.method == 'hmc':
                idata = pm.sample(**self._inference_params.model_dump(exclude={'method', 'n_iterations'}))
            else:  # 'advi'
                approx = pm.fit(n=self._inference_params.n_iterations, method='advi')
                idata = approx.sample(n_samples)
            posterior_predictive = pm.sample_posterior_predictive(idata, var_names=['y_obs', 'A']).posterior_predictive

        # Extract predictions for unobserved months in last year
        logging.info("Extracting predictions...")
        # posterior = idata.posterior
        posterior = posterior_predictive
        predictions_xr = fourier_model.extract_predictions(posterior, model_input)
        assert model_input.y_std is not None and model_input.y_mean is not None
        predictions_xr = predictions_xr*model_input.y_std +model_input.y_mean
        # Select only the first prediction_length months
        predictions_xr = predictions_xr.isel(epi_offset=slice(0, self._prediction_length))

        # Flatten chains and draws: (chain, draw, location, epi_offset) -> (location, epi_offset, samples)
        predictions_samples = predictions_xr.stack(samples=('chain', 'draw')).values

        # Transform back from log space: y = exp(log(y+1)) - 1
        predictions_samples = np.expm1(predictions_samples)

        # Clamp negative values to 0 (disease cases can't be negative)
        predictions_samples = np.maximum(predictions_samples, 0)

        # Create output DataFrame
        predictions_df = create_output(training_data, predictions_samples, n_samples=n_samples, season_length=model_input.season_length)

        if return_inference_data:
            return predictions_df, idata
        else:
            return predictions_df
