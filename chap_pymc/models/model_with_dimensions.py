import dataclasses
from typing import Any, Literal

import numpy as np
import pydantic
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr
from pytensor.xtensor import as_xtensor


@dataclasses.dataclass
class ModelInput:
    X: xr.DataArray
    y: xr.DataArray
    seasonal_pattern: xr.DataArray
    seasonal_errors: xr.DataArray
    last_month: int

class ModelParams(pydantic.BaseModel):
    errors: Literal['iid', 'rw'] = 'rw'
    use_mixture: bool = False
    mixture_weight_prior: tuple[float, float] = (0.5, 0.5)  # U-shaped: heavy at 0 and 1, low in middle

class DimensionalModel:
    def __init__(self, model_params: ModelParams=ModelParams()):
        self._model_params = model_params

    def build_model(self, model_input: ModelInput):
        params = self._model_params
        L, Y, M = model_input.y.shape
        if model_input.X.size:
            eta = self._linear_effect(model_input)
        else:
            eta = 0

        loc_mu = pmd.Normal('loc_mu', mu=0, sigma=10, dims='location')
        loc_sigma = pmd.HalfNormal('loc_sigma', sigma=10)
        loc = pmd.Normal('loc', mu=loc_mu, sigma=loc_sigma, dims=('location', 'year')) + eta
        scale_mu = pmd.Normal('scale_mu', mu=1, sigma=1, dims='location')
        scale_sigma = pmd.HalfNormal('scale_sigma', sigma=1)
        scale = pmd.Normal('scale', scale_mu, sigma=scale_sigma, dims=('location', 'year'))
        seasonal = pmd.as_xtensor(model_input.seasonal_pattern, dims=('location', 'month'))
        transformed_pattern = pmd.Deterministic(
            'transformed_pattern',
            seasonal * scale + loc)

        # Mixture of normal season and empty season

        if params.errors == 'rw':
            epsilon = self._ar_effect(model_input)
        else:
            epsilon = 0

        transformed_samples = pmd.Deterministic('transformed_samples', transformed_pattern + epsilon,
                                                dims=('location', 'year', 'month'))

        # Apply mixture model if enabled
        if params.use_mixture:
            # Continuous mixture weight per location-year: 0 < z < 1
            # z=1 means full seasonal pattern, z=0 means flat line at 0
            alpha, beta = params.mixture_weight_prior
            z = pmd.Beta('mixture_weight',
                         alpha=alpha, beta=beta, dims=('location', 'year'))

            # Mix the means: z * transformed_samples + (1-z) * 0 = z * transformed_samples
            # z will be broadcast to (L, Y, M) automatically
            mu_samples = pmd.Deterministic('mu_mixed',
                                           z * transformed_samples,
                                           dims=('location', 'year', 'month'))
        else:
            mu_samples = transformed_samples

        sigma = pmd.HalfNormal('sigma', sigma=1, dims='location')
        seen_year_samples = mu_samples.isel(year=slice(None, -1))
        seen_year_observed = model_input.y.isel(year=slice(None, -1)).values
        pm.Normal('y_obs',
                  mu=seen_year_samples.values,
                  sigma=sigma.values[:, None, None],
                  observed=seen_year_observed)

        last_year_observed = model_input.y.isel(year=slice(-1, None),
                                                month=slice(None, model_input.last_month + 1)).values
        last_year_mu = mu_samples.isel(year=slice(-1, None), month=slice(None, model_input.last_month + 1))
        pm.Normal('last_year',
                  mu=last_year_mu.values,
                  sigma=sigma.values[:, None, None],
                  observed=last_year_observed)

    def _ar_effect(self, model_input: ModelInput) -> Any:
        L, Y, M = model_input.y.shape
        ar_sigma = pm.HalfNormal('ar_sigma', sigma=0.2, dims='location')

        init_dist = pm.Normal.dist(np.zeros((L, Y)), ar_sigma[..., None])  # Broadcast here

        epsilon = pm.GaussianRandomWalk('epsilon',
                                        init_dist=init_dist,
                                        mu=0,
                                        sigma=ar_sigma[..., None],  # Broadcast here too
                                        steps=M - 1,
                                        dims=('location', 'year', 'month'))
        epsilon = as_xtensor(epsilon, dims=('location', 'year', 'month'))
        return epsilon

    def _linear_effect(self, model_input: ModelInput) -> Any:
        X = model_input.X
        alpha = pmd.Normal('intercept', mu=0, sigma=10, dims='location')  # Should this be global?
        beta = pm.Normal('slope', mu=0, sigma=10, dims='feature')
        tmp = (X.values @ beta[..., None]).squeeze(-1)  # Shape (L, Y)
        tmp = pmd.as_xtensor(tmp, dims=('location', 'year'))
        eta = pmd.Deterministic('eta', alpha + tmp, dims=('location', 'year'))
        return eta


@pytest.fixture()
def model_input():
    L, Y, M = 3, 4, 12
    months = np.arange(M)
    pattern = np.sin(2 * np.pi * months / 12)

    X = xr.DataArray(np.random.rand(L, Y, 3), dims=['location', 'year', 'feature'])

    y_vals = np.zeros((L, Y, M))
    y_vals[0, 0, :] = pattern
    y_vals[0, 1, :] = -pattern
    y = xr.DataArray(y_vals, dims=['location', 'year', 'month'])
    full_pattern  = np.array([pattern for _ in range(L)])
    seasonal_pattern = xr.DataArray(full_pattern, dims=['location', 'month'])
    seasonal_errors = xr.DataArray(np.ones((L, M)), dims=['location', 'month'])
    model_input = ModelInput(X, y, seasonal_pattern, seasonal_errors, last_month=3)
    return model_input

def test_model_with_dimensions(model_input):
    with pm.Model(coords={
        'feature': [f'temp_lag{3-i}'for i in range(3)],
        'location': [f'Loc{i+1}' for i in range(model_input.y.shape[0])],
        'year': [f'202{i}' for i in range(model_input.y.shape[1])],
        'month': list(range(1, 13))
        }):
        DimensionalModel().build_model(model_input)
        #define_stable_model(model_input)
        idata = pm.sample(10, tune=5, chains=1, return_inferencedata=True)
    assert idata is not None

def test_mixture_model(model_input):
    """Test the mixture model with continuous mixture weights (works with ADVI)"""
    params = ModelParams(use_mixture=True, mixture_weight_prior=(2.0, 2.0))
    with pm.Model(coords={
        'feature': [f'temp_lag{3-i}'for i in range(3)],
        'location': [f'Loc{i+1}' for i in range(model_input.y.shape[0])],
        'year': [f'202{i}' for i in range(model_input.y.shape[1])],
        'month': list(range(1, 13))
        }):
        DimensionalModel(model_params=params).build_model(model_input)
        idata = pm.sample(10, tune=5, chains=1, return_inferencedata=True)
    assert idata is not None
    # Check that mixture-specific variables exist
    assert 'mixture_weight' in idata.posterior
    assert 'mu_mixed' in idata.posterior
    # Verify mixture_weight is per location-year
    assert idata.posterior['mixture_weight'].shape[-2:] == (model_input.y.shape[0], model_input.y.shape[1])
