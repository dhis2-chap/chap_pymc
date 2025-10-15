import numpy as np
import pydantic
import pymc.dims as pmd
import pytest
import xarray
import pymc as pm
import pytensor.xtensor as px

from chap_pymc.model_input_creator import ModelInputCreator

class FourierHyperparameters(pydantic.BaseModel):
    periods: int = 12
    n_harmonics: int = 2


class FourierParametrization:

    def __init__(self, hyper_params: FourierHyperparameters = FourierHyperparameters()):
        self.hyper_params = hyper_params

    def get_model(self, y: xarray.DataArray):
        months_xt = pmd.as_xtensor(np.arange(len(y.coords['month'])), dims=('month',))
        # Include h=0 (baseline) as the 0th harmonic
        harmonics_xt = pmd.as_xtensor(np.arange(0, self.hyper_params.n_harmonics + 1), dims=('harmonic',))

        # Use nanmean and nanstd to handle missing data
        global_mean = np.nanmean(y.values)
        global_std = np.nanstd(y.values)

        # Amplitude parameters with harmonic dimension (including h=0 for baseline)
        # For h=0, initialize near global_mean; for h>0, initialize near 0
        harmonics_array = np.arange(0, self.hyper_params.n_harmonics + 1)
        a_mu_init = pmd.as_xtensor(
            np.where(harmonics_array == 0, global_mean, 0.0),
            dims=('harmonic',)
        )
        a_mu = pmd.Normal('a_mu', mu=a_mu_init, sigma=global_std, dims=('location', 'harmonic'))
        a_sigma = pmd.HalfNormal('a_sigma', sigma=global_std, dims=('harmonic',))

        A = pmd.Normal('A', mu=a_mu, sigma=a_sigma, dims=('location', 'year', 'harmonic'))
        phi = pmd.Normal('phi', 0, sigma=np.pi, dims=('location', 'harmonic'))

        # Vectorized frequency calculation
        # h=0: baseline (freq=0, so cos(phi) = constant)
        # h=1: annual cycle (period = 12 months)
        # h=2: semi-annual cycle (period = 6 months)
        freq = 2 * np.pi * harmonics_xt / 12  # Shape: (harmonic,)

        # Broadcasting: months_xt (month,) + phi (location, harmonic)
        # freq (harmonic,) * months_xt (month,) -> (month, harmonic)
        # For h=0: freq=0, so months_phi = phi (constant across months)
        months_phi = freq * months_xt + phi  # (location, harmonic, month) due to broadcasting

        # A is (location, year, harmonic), cos(months_phi) is (location, harmonic, month)
        # For h=0: A[..., 0] * cos(phi[..., 0]) = baseline (constant across months)
        # For h>0: standard harmonic terms
        harmonics_term = A * px.math.cos(months_phi)  # (location, year, harmonic, month)
        mu = pmd.Deterministic('mu', harmonics_term.sum(dim='harmonic'), dims=('location', 'year', 'month'))

        sigma = pm.HalfNormal('sigma', sigma=global_std)
        y_obs = pm.Normal('y_obs', mu=mu.values, sigma=sigma, observed=y)



