from typing import Any

import numpy as np
import pydantic
import pymc.dims as pmd
import xarray
import pymc as pm
import pytensor.xtensor as px
import pytensor.tensor as pt
from pytensor.xtensor.type import XTensorVariable, as_xtensor


from chap_pymc.model_input_creator import  ModelInput

class FourierHyperparameters(pydantic.BaseModel):
    n_harmonics: int = 2
    do_ar_effect: bool = False
    do_mixture: bool = False
    mixture_weight_prior: tuple[float, float] = (0.5, 0.5)  # U-shaped: heavy at 0 and 1, low in middle

class FourierParametrization:

    def __init__(self, hyper_params: FourierHyperparameters = FourierHyperparameters()):
        self.hyper_params = hyper_params

    def get_regression_model(self, X: xarray.DataArray, y: xarray.DataArray):
        return self.get_model(y, A_offset=self._linear_effect(X))

    def _mixture_weights(self, n_years) -> float | pmd.DimDistribution:
        if self.hyper_params.do_mixture:
            # Continuous mixture weight per location-year: 0 < z < 1
            # z=1 means full seasonal pattern, z=0 means flat line at 0
            alpha, beta = self.hyper_params.mixture_weight_prior

            alpha = np.full(n_years, alpha)
            beta = np.full(n_years, beta)

            alpha[-1] = 15.0  # Strong prior towards 1 in last year
            beta[-1] = 1.0
            alpha, beta = (as_xtensor(v, dims=('year',)) for v in (alpha, beta))

            z = pmd.Beta('mixture_weight',
                         alpha=alpha, beta=beta, dims=('location', 'year'))
            return z
        else:
            return 1.0

    def get_model(self, y: xarray.DataArray, A_offset: float | pmd.DimDistribution = 0.0):
        a_mu = pmd.Normal('a_mu', mu=0, sigma=10, dims=('location', 'harmonic'))
        ha_sigma = pmd.HalfNormal('harmonic_a_sigma', sigma=1, dims=('harmonic',))
        a_sigma = pmd.HalfNormal('a_sigma', sigma=ha_sigma, dims=('harmonic',))
        A = pmd.Normal('A', mu=a_mu, sigma=a_sigma, dims=('location', 'year', 'harmonic'))
        A = A + A_offset
        ar_effect = self._ar_effect(y) if self.hyper_params.do_ar_effect else 0
        mu = self._calculate_mu(A, n_months=y.shape[-1]) + ar_effect
        mu = pmd.Deterministic('last_mu', mu * self._mixture_weights(n_years=y.shape[1]))  # Shape: (location, year, month)
        sigma = pmd.HalfNormal('sigma', sigma=1, dims=('location',))

        # Important! Using pmd in the observed statement seems to mess up the inference. Use raw values instead.
        # Careful with broadcasting here: sigma is (location,) and mu is (location, year, month)
        pm.Normal('y_obs', mu=mu.values, sigma=sigma.values[:, None, None], observed=y)

    def _ar_effect(self, y) -> Any:
        L, Y, M = y.shape
        ar_sigma = pm.HalfNormal('ar_sigma', sigma=0.1, dims='location')

        init_dist = pm.Normal.dist(np.zeros((L, Y)), ar_sigma[..., None])  # Broadcast here

        #This is not avaible in pmd, so we use pm directly and then convert to pmd
        epsilon = pm.GaussianRandomWalk('epsilon',
                                        init_dist=init_dist,
                                        mu=0,
                                        sigma=ar_sigma[..., None],  # Broadcast here too
                                        steps=M - 1,
                                        dims=('location', 'year', 'month'))
        epsilon = as_xtensor(epsilon, dims=('location', 'year', 'month'))
        return epsilon

    def _calculate_mu(self, A: pmd.DimDistribution, n_months) -> XTensorVariable:
        months = pmd.as_xtensor(np.arange(n_months), dims=('month',))
        harmonics = pmd.as_xtensor(np.arange(self.hyper_params.n_harmonics + 1), dims=('harmonic',))
        phi = pmd.Normal('phi', 0, sigma=np.pi, dims=('location', 'harmonic'))
        freq = 2 * np.pi * harmonics / 12  # Shape: (harmonic,)
        months_phi = freq * months + phi  # (location, harmonic, month) due to broadcasting
        harmonics_term = A * px.math.cos(months_phi)  # (location, year, harmonic, month)
        mu = pmd.Deterministic('mu', harmonics_term.sum(dim='harmonic'), dims=('location', 'year', 'month'))
        return mu

    def _linear_effect(self, X: xarray.DataArray) -> pmd.DimDistribution:
        beta = pmd.Normal('slope', mu=0, sigma=10, dims=('feature', 'harmonic'))
        # This is a batched matrix multiplication over the feature dimension
        result_tensor = pt.tensordot(X.values, beta.values, axes=[[2], [0]])
        return pmd.Deterministic('linear_effect', result_tensor, dims=('location', 'year', 'harmonic'))

    def extract_predictions(self, idata, model_input: ModelInput) -> xarray.DataArray:
        """
        Extract posterior samples for predicted (unobserved) y values in the last year.

        These are months after model_input.last_month in the final year, which were
        NaN in the observed data and thus imputed by PyMC during sampling.

        Args:
            idata: ArviZ InferenceData with posterior samples
            model_input: ModelInput with y data and last_month information

        Returns:
            xarray.DataArray with dims (chain, draw, location, month) containing
            posterior samples for the prediction months
        """
        # Get the full y_obs samples from posterior (includes both observed and imputed)
        y_samples = idata.posterior['y_obs']  # (chain, draw, location, year, month)

        # Extract last year
        last_year_idx = model_input.y.shape[1] - 1 if isinstance(model_input.y, np.ndarray) else len(model_input.y.coords['year']) - 1
        y_last_year = y_samples.isel(y_obs_dim_1=last_year_idx)  # (chain, draw, location, month)

        # Filter for prediction months (after last_month)
        # last_month is 0-indexed, so months to predict are last_month+1 onwards
        prediction_months_idx = np.arange(model_input.last_month + 1, model_input.n_months())
        y_predictions = y_last_year.isel(y_obs_dim_2=prediction_months_idx)

        # Rename dimensions to be more intuitive
        y_predictions = y_predictions.rename({
            'y_obs_dim_0': 'location',
            'y_obs_dim_2': 'month'
        })

        # Add proper month coordinates if available
        if hasattr(model_input.y, 'coords') and 'month' in model_input.y.coords:
            month_labels = model_input.y.coords['month'].values[prediction_months_idx]
            y_predictions = y_predictions.assign_coords(month=month_labels)

        if hasattr(model_input.y, 'coords') and 'location' in model_input.y.coords:
            location_labels = model_input.y.coords['location'].values
            y_predictions = y_predictions.assign_coords(location=location_labels)

        return y_predictions



