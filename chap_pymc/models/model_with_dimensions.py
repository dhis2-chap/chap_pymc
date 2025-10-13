import dataclasses
from typing import Literal

import numpy as np
import pydantic
import xarray as xr
import pymc as pm
import pymc.dims as pmd
import pytest
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

def define_stable_model(model_input: ModelInput, params=ModelParams()):
    L, Y, M = model_input.y.shape
    if model_input.X.size:
        X = model_input.X
        alpha = pmd.Normal('intercept', mu=0, sigma=10, dims='location')  # Should this be global?
        beta = pm.Normal('slope', mu=0, sigma=10, dims='feature')
        tmp = (X.values @ beta[..., None]).squeeze(-1)  # Shape (L, Y)
        tmp = pmd.as_xtensor(tmp, dims=('location', 'year'))
        eta = pmd.Deterministic('eta', alpha + tmp, dims=('location', 'year'))
    else:
        eta = 0

    loc_mu = pmd.Normal('loc_mu', mu=0, sigma=10, dims='location')
    loc_sigma = pmd.HalfNormal('loc_sigma', sigma=10)
    loc = pmd.Normal('loc', mu=loc_mu, sigma=loc_sigma, dims=('location', 'year')) + eta
    scale_mu = pmd.Normal('scale_mu', mu=1, sigma=1, dims='location')
    scale_sigma = pmd.HalfNormal('scale_sigma', sigma=1)
    scale = pmd.Normal('scale', scale_mu, sigma=scale_sigma, dims=('location', 'year'))

    #transformed_pattern = pm.Deterministic(
    #'transformed_pattern',
    #model_input.seasonal_pattern.values[:, None, :] * scale[..., None] + loc[..., None],
    seasonal = pmd.as_xtensor(model_input.seasonal_pattern, dims=('location', 'month'))
    transformed_pattern=pmd.Deterministic(
            'transformed_pattern',
        seasonal * scale + loc)


    if params.errors == 'rw':
        ar_sigma = pm.HalfNormal('ar_sigma', sigma=0.2, dims='location')

        init_dist = pm.Normal.dist(np.zeros((L, Y)), ar_sigma[..., None])  # Broadcast here

        epsilon = pm.GaussianRandomWalk('epsilon',
                                        init_dist=init_dist,
                                        mu=0,
                                        sigma=ar_sigma[..., None],  # Broadcast here too
                                        steps=M - 1,
                                        dims=('location', 'year', 'month'))
        epsilon = as_xtensor(epsilon, dims=('location', 'year', 'month'))
    else:
        epsilon = 0

    transformed_samples = pmd.Deterministic('transformed_samples', transformed_pattern + epsilon, dims=('location', 'year', 'month'))
    sigma = pmd.HalfNormal('sigma', sigma=1, dims='location')
    seen_year_samples = transformed_samples.isel(year=slice(None, -1))
    seen_year_observed = model_input.y.isel(year=slice(None, -1)).values
    pm.Normal('y_obs',
               mu=seen_year_samples.values,
               sigma=sigma.values[:, None, None],
               observed=seen_year_observed)

    last_year_observed = model_input.y.isel(year=slice(-1, None), month=slice(None, model_input.last_month + 1)).values
    #[:, -1:, :model_input.laast_month + 1]
    last_year_mu = transformed_samples.isel(year=slice(-1, None), month=slice(None, model_input.last_month + 1))
    pm.Normal('last_year',
               mu=last_year_mu.values,
               sigma=sigma.values[:, None, None],
               observed=last_year_observed)


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
        }) as model:
        define_stable_model(model_input)
        idata = pm.sample(10, tune=5, chains=1, return_inferencedata=True)
    assert idata is not None