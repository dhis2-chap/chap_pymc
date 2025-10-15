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
        harmonics_xt = pmd.as_xtensor(np.arange(1, self.hyper_params.n_harmonics + 1), dims=('harmonic',))
        # Use nanmean and nanstd to handle missing data
        global_mean = np.nanmean(y.values)
        global_std = np.nanstd(y.values)
        location_baseline = pmd.Normal(
            'baseline_mu', mu=global_mean, sigma=global_std, dims=('location',))

        baseline = pmd.Normal('baseline', 
                              mu=location_baseline,
                              sigma=global_std,
                              dims=('location', 'year'))

        # Amplitude parameters with harmonic dimension
        a_mu = pmd.Normal('a_mu', mu=0, sigma=global_std, dims=('location', 'harmonic'))
        a_sigma = pmd.HalfNormal('a_sigma', sigma=global_std, dims=('harmonic',))

        A = pmd.Normal('A', mu=a_mu, sigma=a_sigma, dims=('location', 'year', 'harmonic'))
        phi = pmd.Normal('phi', 0, sigma=np.pi, dims=('location', 'harmonic'))

        # Vectorized frequency calculation
        # h=1: annual cycle (period = 12 months)
        # h=2: semi-annual cycle (period = 6 months)
        freq = 2 * np.pi * harmonics_xt / 12  # Shape: (harmonic,)

        # Broadcasting: months_xt (month,) + phi (location, harmonic)
        # Result needs to be (location, month, harmonic) to work with freq
        # freq (harmonic,) * months_xt (month,) -> (month, harmonic)
        months_phi = freq * months_xt + phi  # (location, harmonic, month) due to broadcasting

        # A is (location, year, harmonic), cos(months_phi) is (location, harmonic, month)
        # Need to sum over harmonic dimension
        harmonics_term = A * px.math.cos(months_phi)  # (location, year, harmonic, month)
        harmonics_sum = harmonics_term.sum(dim='harmonic')  # Sum over harmonic dimension -> (location, year, month)

        mu = pmd.Deterministic('mu', baseline + harmonics_sum, dims=('location', 'year', 'month'))
        sigma = pm.HalfNormal('sigma', sigma=global_std)
        y_obs = pm.Normal('y_obs', mu=mu.values, sigma=sigma, observed=y)



