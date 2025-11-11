import logging
from typing import Any

import numpy as np
import pydantic
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytensor.xtensor as px
import xarray
from pytensor.xtensor.type import as_xtensor, XTensorVariable

from chap_pymc.transformations.model_input_creator import FourierModelInput

logger = logging.getLogger(__name__)


class FourierHyperparameters(pydantic.BaseModel):
    n_harmonics: int = 3
    do_ar_effect: bool = False
    do_mixture: bool = False
    mixture_weight_prior: tuple[float, float] = (0.5, 0.5)  # U-shaped: heavy at 0 and 1, low in middle
    use_prev_year: bool = True

class FourierParametrization:

    def __init__(self, hyper_params: FourierHyperparameters = FourierHyperparameters()) -> None:
        self.hyper_params = hyper_params

    @property
    def extra_dims(self) -> dict[str, Any]:
        '''Coordinates for extra dimensions used in the model.'''
        return {'harmonic': np.arange(0, self.hyper_params.n_harmonics + 1)}  # Include baseline (h=0)

    def get_regression_model(self, X: xarray.DataArray, y: xarray.DataArray, prev_year_y: xarray.DataArray | None = None) -> None:
        return self.get_model(y, prev_year_y=prev_year_y, A_offset=self._linear_effect(X))

    def _mixture_weights(self, n_years: int) -> float | Any:
        if self.hyper_params.do_mixture:
            # Continuous mixture weight per location-year: 0 < z < 1
            # z=1 means full seasonal pattern, z=0 means flat line at 0
            alpha_val, beta_val = self.hyper_params.mixture_weight_prior

            alpha_arr = np.full(n_years, alpha_val)
            beta_arr = np.full(n_years, beta_val)

            alpha_arr[-1] = 15.0  # Strong prior towards 1 in last year
            beta_arr[-1] = 1.0
            alpha, beta = (as_xtensor(v, dims=('epi_year',)) for v in (alpha_arr, beta_arr))

            z = pmd.Beta('mixture_weight',
                         alpha=alpha, beta=beta, dims=('location', 'epi_year'))
            return z
        else:
            return 1.0

    def _get_mv_harmonic(self, dim: str, coords: dict[str, Any]) -> Any:
        pm.modelcontext(None).add_coord(f'{dim}_corr', coords[dim])
        n = len(coords[dim])
        sd_dist = pm.Exponential.dist(1.0, size=n)
        cholesky, *_ = pm.LKJCholeskyCov('cholesky_raw',
                                         eta=2, n=n, sd_dist=sd_dist,
                                         compute_corr=True)
        chol = pmd.as_xtensor(cholesky, dims=(dim, f'{dim}_corr'), name='cholesky')
        h_mu = pmd.Normal('h_{dim}_mu', mu=0, sigma=1, dims=(dim,))
        mv = pmd.MvNormal(f'{dim}_mu', mu=h_mu, chol=chol, core_dims=(dim, f'{dim}_corr'), dims=('location', dim))
        return mv

    def get_model(self, y: xarray.DataArray, prev_year_y: xarray.DataArray | None = None, A_offset: float | Any = 0.0) -> None:
        # Non-centered parametrization to avoid funnel geometry
        a_mu = pmd.Normal('a_mu', mu=0, sigma=1, dims=('location', 'harmonic'))
        a_sigma = pmd.HalfNormal('a_sigma', sigma=1., dims=('harmonic',))
        l_sigma = pmd.HalfNormal('l_sigma', sigma=1., dims=('location',))
        A_raw = pmd.Normal('A_raw', mu=0, sigma=1, dims=('location', 'epi_year', 'harmonic'))
        A = pmd.Deterministic('A', a_mu + A_raw * a_sigma * l_sigma, dims=('location', 'epi_year', 'harmonic'))
        s = A + A_offset
        ar_effect = self._ar_effect(y) if self.hyper_params.do_ar_effect else 0
        mu = self._calculate_mu(s, n_months=y.shape[-1]) + ar_effect
        mu = pmd.Deterministic('last_mu',
                               mu * self._mixture_weights(n_years=y.sizes['epi_year']),
                               dims=('location', 'epi_year', 'epi_offset'))
                               # Shape: (location, epi_year, epi_offset)
        # Hierarchical observation noise
        sigma_scale = pmd.HalfNormal('sigma_scale', sigma=0.5)
        sigma = pmd.HalfNormal('sigma', sigma=sigma_scale)

        # Important! Using pmd in the observed statement seems to mess up the inference. Use raw values instead.
        # Careful with broadcasting here: sigma is (location,) and mu is (location, epi_year, epi_offset)

        self._get_flat_obs_model(mu, sigma, y)
        #y_raw = pmd.Normal('y_raw', mu=0, sigma=1, dims=('location', 'epi_year', 'epi_offset'))
        pmd.Normal('y_obs', mu=mu, sigma=sigma, dims=('location', 'epi_year', 'epi_offset'))
        #pm.Deterministic('y_obs', y_raw * sigma + mu)
        #pm.Normal('y_obs', mu=mu.values, sigma=sigma.values[:, None, None], observed=y, dims=('location', 'epi_year', 'epi_offset'))

        # Previous year observation (if enabled)
        if self.hyper_params.use_prev_year and prev_year_y is not None:
            # Location-specific offset and noise parameters
            offset_loc = pmd.Normal('offset_loc', mu=0, sigma=1, dims='location')
            sigma_loc = pmd.HalfNormal('sigma_loc', sigma=0.5)

            # Expected value: first month prediction (epi_offset=0) for each year
            mu_first_month = mu.values[..., 0]

            # Observation: prev_year_y = offset_loc + Normal(0, sigma_loc) + mu_first_month
            pm.Normal('y_prev_year_obs',
                       mu=mu_first_month + offset_loc.values[:, None],
                       sigma=sigma_loc.values,
                       observed=prev_year_y.values,
                       dims=('location', 'epi_year'))

    def _get_flat_obs_model(self, mu: XTensorVariable, sigma: pm.HalfNormal, y: xarray.DataArray):
        missing_mask = y.isnull()
        mv = missing_mask.values
        flat_mu = mu.values[~mv]
        flat_sigma = sigma.values
        # flat_sigma = pm.math.broadcast_to(sigma.values[:, None, None], mu.values.shape)[~mv]
        pm.Normal('flat_observed', flat_mu, sigma=flat_sigma, observed=y.values[~mv])

    def _ar_effect(self, y: xarray.DataArray) -> Any:
        L, Y, M = y.shape
        ar_sigma = pm.HalfNormal('ar_sigma', sigma=0.1, dims='location')

        init_dist = pm.Normal.dist(np.zeros((L, Y)), ar_sigma[..., None])  # Broadcast here

        #This is not avaible in pmd, so we use pm directly and then convert to pmd
        epsilon = pm.GaussianRandomWalk('epsilon',
                                        init_dist=init_dist,
                                        mu=0,
                                        sigma=ar_sigma[..., None],  # Broadcast here too
                                        steps=M - 1,
                                        dims=('location', 'epi_year', 'epi_offset'))
        epsilon = as_xtensor(epsilon, dims=('location', 'epi_year', 'epi_offset'))
        return epsilon

    def _calculate_mu(self, A: Any, n_months: int) -> Any:
        months = pmd.as_xtensor(np.arange(n_months), dims=('epi_offset',))
        harmonics = pmd.as_xtensor(np.arange(self.hyper_params.n_harmonics + 1), dims=('harmonic',))
        phi = pmd.Normal('phi', 0, sigma=np.pi, dims=('location', 'harmonic'))
        freq = 2 * np.pi * harmonics / 12  # Shape: (harmonic,)
        months_phi = freq * months + phi  # (location, harmonic, epi_offset) due to broadcasting
        harmonics_term = A * px.math.cos(months_phi)  # (location, epi_year, harmonic, epi_offset)
        mu = pmd.Deterministic('mu', harmonics_term.sum(dim='harmonic'), dims=('location', 'epi_year', 'epi_offset'))
        return mu

    def _linear_effect(self, X: xarray.DataArray) -> Any:
        beta = pmd.Normal('slope', mu=0, sigma=10, dims=('feature', 'harmonic'))
        # This is a batched matrix multiplication over the feature dimension
        result_tensor = pt.tensordot(X.values, beta.values, axes=[[2], [0]])
        return pmd.Deterministic('linear_effect', result_tensor, dims=('location', 'epi_year', 'harmonic'))

    def extract_predictions(self, posterior: Any, model_input: FourierModelInput) -> xarray.DataArray:
        """
        Extract posterior samples for predicted (unobserved) y values in the last year.

        These are months after model_input.last_month in the final year, which were
        NaN in the observed data and thus imputed by PyMC during sampling.

        Args:
            idata: ArviZ InferenceData with posterior samples
            model_input: ModelInput with y data and last_month information

        Returns:
            xarray.DataArray with dims (chain, draw, location, epi_offset) containing
            posterior samples for the prediction months
        """
        # Get the full y_obs samples from posterior (includes both observed and imputed)
        nans = model_input.y.isnull()
        logger.info('NANS: -------')
        logger.info(
            nans.any(dim=('location', 'epi_year')))
        logger.info(nans.any(dim=('location', 'epi_offset')))


        try:
            y_samples: xarray.DataArray = posterior['y_obs']  # (chain, draw, location, epi_year, epi_offset)
        except Exception:
            for key in posterior.data_vars:
                logger.error("Posterior variable: %s with dims %s", key, posterior[key].dims)
            raise

        # Extract last year
        #last_year_idx = model_input.y.shape[1] - 1 if isinstance(model_input.y, np.ndarray) else len(model_input.y.coords['epi_year']) - 1
        y_last_year = y_samples.isel(epi_year=(-1-model_input.added_last_year))
        prediction_months_idx = np.arange(model_input.last_month + 1, model_input.n_months())
        y_predictions = y_last_year.isel(epi_offset=prediction_months_idx)
        if model_input.added_last_year:
            final_last_year = y_samples.isel(epi_year=-1)
            y_predictions = xarray.concat((y_predictions, final_last_year), dim='epi_offset')
            logger.info(y_predictions.coords)

        # Filter for prediction months (after last_month)
        # last_month is 0-indexed, so months to predict are last_month+1 onwards

        if len(prediction_months_idx) < 3:
            logger.info(f"Only {len(prediction_months_idx)} months to predict, expected 3.")
            logger.info(f'model_input.last_month: {model_input.last_month}')
            logger.info(f'y_coords epi_offset: {model_input.y.coords}')
            logger.info(f'y_pred: {y_samples.coords}')


        return y_predictions



