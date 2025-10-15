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
        months = np.arange(0, self.hyper_params.periods)
        # Convert months to xtensor for proper broadcasting
        months_xt = pmd.as_xtensor(months, dims=('month',))

        n_harmonics = self.hyper_params.n_harmonics
        # Use nanmean and nanstd to handle missing data
        global_mean = np.nanmean(y.values)
        global_std = np.nanstd(y.values)
        location_baseline = pmd.Normal(
            'baseline_mu', mu=global_mean, sigma=global_std, dims=('location',))

        baseline = pmd.Normal('baseline',
                              mu=location_baseline,
                              sigma=global_std,
                              dims=('location', 'year'))
        # For each harmonic
        harmonics_sum = 0

        for h in range(1, n_harmonics + 1):
            # Amplitude (constrained to be positive)
            a_mu = pmd.Normal(f'a_mu{h}', mu=0, sigma=global_std, dims=('location',))
            a_sigma = pmd.HalfNormal(f'a_sigma{h}', sigma=global_std)
            A = pmd.Normal(f'A{h}',mu=a_mu, sigma=a_sigma, dims=('location', 'year'))

            # Phase shift (in radians, 0 to 2π)
            phi = pmd.Normal(f'phi{h}', 0, sigma=np.pi, dims=('location',))
            # Add harmonic component
            # h=1: annual cycle (period = 12 months)
            # h=2: semi-annual cycle (period = 6 months)
            freq = 2 * np.pi * h / 12
            months_phi = freq * months_xt + phi

            harmonics_sum += A * px.math.cos(months_phi)

        mu = pmd.Deterministic('mu', baseline + harmonics_sum, dims=('location', 'year', 'month'))
        sigma = pm.HalfNormal('sigma', sigma=global_std)
        y_obs = pm.Normal('y_obs', mu=mu.values, sigma=sigma, observed=y)



