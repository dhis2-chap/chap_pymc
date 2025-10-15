from typing import Any

import numpy as np
import pydantic
import pymc.dims as pmd
import pytest
import xarray
import pymc as pm
import pytensor.xtensor as px
import pytensor.tensor as pt
from pytensor.xtensor.type import XTensorConstant, XTensorType, XTensorVariable
from xarray import Variable

from chap_pymc.model_input_creator import ModelInputCreator

class FourierHyperparameters(pydantic.BaseModel):
    periods: int = 12
    n_harmonics: int = 2


class FourierParametrization:

    def __init__(self, hyper_params: FourierHyperparameters = FourierHyperparameters()):
        self.hyper_params = hyper_params

    def get_regression_model(self, X: xarray.DataArray, y: xarray.DataArray):
        return self.get_model(y, A_offset=self._linear_effect(X))

    def get_model(self, y: xarray.DataArray, A_offset: float | pmd.DimDistribution = 0.0):
        months = pmd.as_xtensor(np.arange(len(y.coords['month'])), dims=('month',))
        a_mu = pmd.Normal('a_mu', mu=0, sigma=10, dims=('location', 'harmonic'))
        a_sigma = pmd.HalfNormal('a_sigma', sigma=10, dims=('harmonic',))
        A = pmd.Normal('A', mu=a_mu, sigma=a_sigma, dims=('location', 'year', 'harmonic'))
        A = A + A_offset
        mu = self._calculate_mu(A, months)
        sigma = pm.HalfNormal('sigma', sigma=10)
        pm.Normal('y_obs', mu=mu.values, sigma=sigma, observed=y)

    def _calculate_mu(self, A: float, months: Variable | XTensorConstant[XTensorType | Any] | Any) -> XTensorVariable:
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



