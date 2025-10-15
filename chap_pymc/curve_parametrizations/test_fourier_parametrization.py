import pymc as pm
import numpy as np
import pytest
import xarray
from chap_pymc.curve_parametrizations.fourier_parametrization_plots import plot_vietnam_faceted_predictions
from chap_pymc.curve_parametrizations.fourier_parametrization import FourierParametrization, FourierHyperparameters
from chap_pymc.model_input_creator import ModelInputCreator


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


@pytest.fixture()
def viet_model_input(viet_begin_season):
    creator = ModelInputCreator(prediction_length=3, lag=3, mask_empty_seasons=False)
    model_input = creator.create_model_input(viet_begin_season)
    return creator.to_xarray(model_input)


@pytest.fixture
def vietnam_y_xarray(viet_model_input):
    """Get y-xarray for Vietnam dataset"""
    return viet_model_input.y


def test_vietnam_y_xarray_fixture(vietnam_y_xarray):
    """Test that the Vietnam y-xarray fixture works correctly"""
    # Check that it's an xarray DataArray
    assert isinstance(vietnam_y_xarray, xarray.DataArray)

    # Check that it has the expected dimensions
    assert vietnam_y_xarray.dims == ('location', 'year', 'month')

    # Check that it has coordinates
    assert 'location' in vietnam_y_xarray.coords
    assert 'year' in vietnam_y_xarray.coords
    assert 'month' in vietnam_y_xarray.coords

    # Check shape makes sense
    assert vietnam_y_xarray.shape[2] == 12  # 12 months

    print(f"Vietnam y-xarray shape: {vietnam_y_xarray.shape}")
    print(f"Dimensions: {vietnam_y_xarray.dims}")
    print(f"Locations: {list(vietnam_y_xarray.coords['location'].values)}")
    print(f"Years: {list(vietnam_y_xarray.coords['year'].values)}")


def test_fourier_parametrization(y, coords):
    """Test Fourier parametrization with synthetic data and create faceted plot"""
    from chap_pymc.curve_parametrizations.fourier_parametrization_plots import plot_faceted_predictions
    coords |= {'harmonic': np.arange(0, 3)}  # Add harmonic coordinate (0=baseline, 1-2=harmonics)
    with pm.Model(coords=coords) as model:
        FourierParametrization().get_model(y)
        idata = pm.sample(draws=100, tune=100, progressbar=True, return_inferencedata=True)

    # Extract posterior predictions
    mu_posterior = idata.posterior['mu']  # (chain, draw, location, year, month)
    mu_mean = mu_posterior.mean(dim=['chain', 'draw'])  # (location, year, month)
    mu_lower = mu_posterior.quantile(0.025, dim=['chain', 'draw'])
    mu_upper = mu_posterior.quantile(0.975, dim=['chain', 'draw'])

    # Create plot using plotting function
    plot_faceted_predictions(y, mu_mean, mu_lower, mu_upper, coords)

@pytest.fixture
def viet_coords(viet_model_input):
    n_harmonics = 3  # Number of oscillating harmonics (not counting baseline)
    return {
        'location': viet_model_input.y.coords['location'].values,
        'year': viet_model_input.y.coords['year'].values,
        'month': viet_model_input.y.coords['month'].values,
        'harmonic': np.arange(0, n_harmonics + 1),  # [0, 1, 2, 3]: 0=baseline, 1-3=oscillating
        'feature': viet_model_input.X.coords['feature'].values
    }


def test_vietnam_regression(viet_model_input, viet_coords):
    n_harmonics = len(viet_coords['harmonic']) - 1  # Subtract 1 for baseline
    with pm.Model(coords=viet_coords) as model:
        m = FourierParametrization(FourierHyperparameters(n_harmonics=n_harmonics))
        m.get_regression_model(viet_model_input.X, viet_model_input.y)
        pm.model_to_graphviz(model).render('fourier_graph', format='png', view=True)
        idata = pm.sample(draws=500, tune=500, progressbar=True, return_inferencedata=True)
        az.plot_posterior(idata, var_names=['slope', 'a_mu', 'a_sigma', 'sigma'])
    # Extract posterior predictions
    mu_posterior = idata.posterior['mu']  # (chain, draw, location, year, month)
    mu_mean = mu_posterior.mean(dim=['chain', 'draw'])  # (location, year, month)
    mu_lower = mu_posterior.quantile(0.025, dim=['chain', 'draw'])
    mu_upper = mu_posterior.quantile(0.975, dim=['chain', 'draw'])

    # Create plot using plotting function
    plot_vietnam_faceted_predictions(viet_model_input.y, mu_mean, mu_lower, mu_upper, viet_coords)


def test_vietnam_fourier_fit(vietnam_y_xarray):
    """Fit Fourier parametrization to Vietnam dataset and plot results"""

    n_harmonics = 3
    # Extract coordinates from the data
    coords = {
        'location': vietnam_y_xarray.coords['location'].values,
        'year': vietnam_y_xarray.coords['year'].values,
        'month': vietnam_y_xarray.coords['month'].values,
        'harmonic': np.arange(0, n_harmonics+1)  # Add harmonic coordinate (0=baseline, 1-n=harmonics)
    }
    # Build and sample the model
    with pm.Model(coords=coords) as model:
        FourierParametrization(FourierHyperparameters(n_harmonics=n_harmonics)).get_model(vietnam_y_xarray)
        pm.model_to_graphviz(model).render('fourier_graph', format='png', view=True)
        idata = pm.sample(draws=500, tune=500, progressbar=True, return_inferencedata=True)



    # Extract posterior predictions
    mu_posterior = idata.posterior['mu']  # (chain, draw, location, year, month)
    mu_mean = mu_posterior.mean(dim=['chain', 'draw'])  # (location, year, month)
    mu_lower = mu_posterior.quantile(0.025, dim=['chain', 'draw'])
    mu_upper = mu_posterior.quantile(0.975, dim=['chain', 'draw'])

    # Create plot using plotting function
    plot_vietnam_faceted_predictions(vietnam_y_xarray, mu_mean, mu_lower, mu_upper, coords)


def test_vietnam_parameter_correlations(viet_model_input):
    """Fit Fourier model and plot parameter-feature correlations"""
    from chap_pymc.curve_parametrizations.fourier_parametrization_plots import plot_parameter_feature_correlations

    # Extract coordinates and data
    vietnam_y = viet_model_input.y
    n_harmonics = 3
    coords = {
        'location': vietnam_y.coords['location'].values,
        'year': vietnam_y.coords['year'].values,
        'month': vietnam_y.coords['month'].values,
        'harmonic': np.arange(0, n_harmonics+1)  # Add harmonic coordinate (0=baseline, 1-n=harmonics)
    }



    # Build and sample the model
    with pm.Model(coords=coords) as model:
        FourierParametrization(FourierHyperparameters(n_harmonics=n_harmonics)).get_model(vietnam_y)
        idata = pm.sample(draws=500, tune=500, progressbar=True, return_inferencedata=True)

    # Create correlation plot
    plot_parameter_feature_correlations(idata, viet_model_input, n_harmonics,
                                       output_file='vietnam_parameter_correlations.html')
