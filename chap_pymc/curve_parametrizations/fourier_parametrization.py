import numpy as np
import pydantic
import pymc.dims as pmd
import pytest
import xarray
import pymc as pm
import pytensor.xtensor as px
import arviz as az
import matplotlib.pyplot as plt

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
        global_mean = np.mean(y.values)
        global_std = np.std(y.values)
        baseline = pmd.Normal('baseline', mu=global_mean, sigma=global_std, dims=('location', 'year'))
        # For each harmonic
        harmonics_sum = 0
        for h in range(1, n_harmonics + 1):
            # Amplitude (constrained to be positive)
            A = pmd.HalfNormal(f'A{h}', sigma=global_std, dims=('location', 'year'))

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

@pytest.fixture()
def coords():
    return {
        'location': np.arange(3),
        'year': np.arange(4),
        'month': np.arange(12)
    }


@pytest.fixture
def y(coords):
    '''
    DataArray With dimensions (location, year, month)
    locations: 3
    years: 4 (2020, 2021, 2022)
    months: 12
    '''
    L, Y, M = len(coords['location']), len(coords['year']), len(coords['month'])
    months = np.arange(M)
    locs = np.arange(L)
    years = np.arange(Y)
    pattern = np.sin(2 * np.pi * months / 12)
    y = locs[:, None, None] + years[None, :, None] + pattern[None, None, :]
    y = xarray.DataArray(y, dims=['location', 'year', 'month'])
    return y

def test_fourier_parametrization(y, coords):
    with pm.Model(coords=coords) as model:
        FourierParametrization().get_model(y)
        idata = pm.sample(draws=100, tune=100, progressbar=True, return_inferencedata=True)

    # Extract posterior predictions
    mu_posterior = idata.posterior['mu']  # (chain, draw, location, year, month)
    mu_mean = mu_posterior.mean(dim=['chain', 'draw'])  # (location, year, month)
    mu_lower = mu_posterior.quantile(0.025, dim=['chain', 'draw'])
    mu_upper = mu_posterior.quantile(0.975, dim=['chain', 'draw'])

    # Create plot for each location
    n_locations = len(coords['location'])
    fig, axes = plt.subplots(n_locations, 1, figsize=(12, 4 * n_locations))
    if n_locations == 1:
        axes = [axes]

    for loc_idx in range(n_locations):
        ax = axes[loc_idx]

        # Plot each year for this location
        n_years = len(coords['year'])
        for year_idx in range(n_years):
            months = np.arange(12)

            # Observed data
            y_obs = y.values[loc_idx, year_idx, :]
            ax.plot(months, y_obs, 'o-', alpha=0.7, label=f'Observed Year {year_idx}')

            # Posterior mean
            y_pred = mu_mean.values[loc_idx, year_idx, :]
            ax.plot(months, y_pred, '--', alpha=0.7, label=f'Predicted Year {year_idx}')

            # Credible interval
            lower = mu_lower.values[loc_idx, year_idx, :]
            upper = mu_upper.values[loc_idx, year_idx, :]
            ax.fill_between(months, lower, upper, alpha=0.2)

        ax.set_xlabel('Month')
        ax.set_ylabel('Value')
        ax.set_title(f'Location {loc_idx}: Observed vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fourier_parametrization_fit.png', dpi=150)
    print("Plot saved to fourier_parametrization_fit.png")
