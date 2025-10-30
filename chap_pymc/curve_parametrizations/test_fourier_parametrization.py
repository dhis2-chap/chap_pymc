import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import numpy as np
import pytest
import xarray
from chap_pymc.curve_parametrizations.fourier_parametrization_plots import plot_vietnam_faceted_predictions
from chap_pymc.curve_parametrizations.fourier_parametrization import FourierParametrization, FourierHyperparameters
from chap_pymc.inference_params import InferenceParams
from chap_pymc.model_input_creator import ModelInputCreator
from chap_pymc.models.seasonal_fourier_regression import SeasonalFourierRegression


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
    creator = ModelInputCreator(prediction_length=3, lag=3)
    model_input = creator.create_model_input(viet_begin_season)
    return model_input

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
@pytest.fixture
def viet_idata_path():
    return 'vietnam_fourier_parametrization_fit.nc'

@pytest.fixture()
def nepal_model_input(nepal_data):
    creator = ModelInputCreator(prediction_length=3, lag=3, mask_empty_seasons=False)
    model_input = creator.create_model_input(nepal_data)
    return creator.to_xarray(model_input)

def test_nepal_regresion(nepal_data):
    fourier_regression = SeasonalFourierRegression(
        prediction_length=3,
        lag=3,
        fourier_hyperparameters=FourierHyperparameters(n_harmonics=2, do_mixture=True),
        inference_params=InferenceParams(method='advi', n_iterations=100_000, progressbar=True)
    )
    preds, idata = fourier_regression.predict(nepal_data,  return_inference_data=True)
    mu_posterior = idata.posterior['last_mu']  # (chain, draw, location, year, month)
    mu_mean = mu_posterior.mean(dim=['chain', 'draw'])  # (location, year, month)
    mu_lower = mu_posterior.quantile(0.025, dim=['chain', 'draw'])
    mu_upper = mu_posterior.quantile(0.975, dim=['chain', 'draw'])

    # Create plot using plotting function
    plot_vietnam_faceted_predictions(fourier_regression.model_input.y, mu_mean, mu_lower, mu_upper, fourier_regression.stored_coords, output_file='nepal_fourier_fit.png')

def test_full_vietnam_regression(viet_full_year):
    for i, (viet_instance, t) in enumerate(viet_full_year):
        if i<7:
            continue
        creator = ModelInputCreator(prediction_length=3, lag=3)
        model_input = creator.create_model_input(viet_instance)
        test_vietnam_regression(model_input, viet_idata_path=f'viet_{i}.nc', i=i)

def test_vietnam_regression(viet_model_input,  viet_idata_path=None, i=0):
    viet_coords= viet_model_input.coords() | {'harmonic': np.arange(0, 4)}  # Add harmonic coordinate (0=baseline, 1-3=harmonics)
    n_harmonics = len(viet_coords['harmonic']) - 1  # Subtract 1 for baseline
    with pm.Model(coords=viet_coords) as model:
        m = FourierParametrization(FourierHyperparameters(n_harmonics=n_harmonics))
        m.get_regression_model(viet_model_input.X, viet_model_input.y)

        pm.model_to_graphviz(model).render('fourier_graph', format='png', view=True)
        if False:
            approx = pm.fit(n=100000, method='advi')

            # Sample from approximation
            idata = approx.sample(1000)
        else:
            idata = pm.sample(draws=500, tune=500, progressbar=True, return_inferencedata=True)
        posterior = pm.sample_posterior_predictive(idata, var_names=['y_obs']).posterior_predictive
        #idata = pm.sample(draws=500, tune=500, chains=4, progressbar=True, return_inferencedata=True)
#az.plot_posterior(idata, var_names=['slope'])
        az.plot_posterior(idata, var_names=['sigma'])
        plt.show()
    # Save idata for inspection
    if viet_idata_path:
        idata.to_netcdf(viet_idata_path)
        viet_model_input.y.to_netcdf(f'y_{viet_idata_path}')
        viet_model_input.X.to_netcdf(f'X_{viet_idata_path}')

    #posterior = idata.posterior
    #posterior['A'].isel(harmonic=0).median(dim=['chain', 'draw']).plot()
    # Extract posterior predictions
    mu_posterior = posterior['y_obs']  # (chain, draw, location, year, month)
    mu_mean = mu_posterior.mean(dim=['chain', 'draw'])  # (location, year, month)
    mu_lower = mu_posterior.quantile(0.025, dim=['chain', 'draw'])
    mu_upper = mu_posterior.quantile(0.975, dim=['chain', 'draw'])

    # Create plot using plotting function
    plot_vietnam_faceted_predictions(viet_model_input.y, mu_mean, mu_lower, mu_upper, viet_coords, output_file=f'vietnam_fourier_fit_{i}.png')

def test_vietnam(viet_begin_season, debug_model):
    preds = debug_model.predict(viet_begin_season)


def test_nepal(nepal_data: pd.DataFrame, debug_model):
    global TESTING
    TESTING = True
    preds = debug_model.predict(nepal_data)

@pytest.fixture
def debug_model() -> SeasonalFourierRegression:
    return SeasonalFourierRegression(
        prediction_length=3,
        lag=3,
        fourier_hyperparameters=FourierHyperparameters(n_harmonics=2, do_mixture=True),
        inference_params=InferenceParams(method='advi', n_iterations=1_000, progressbar=True)
    )


def test_extract_samples(viet_idata_path):
    """Test that we can reload idata and extract samples"""
    idata = az.from_netcdf(viet_idata_path)
    print(idata)
    slope_samples = idata.posterior['slope']
    print(f"Slope samples shape: {slope_samples.shape}")
    assert slope_samples.shape[0] > 0  # Chains
    assert slope_samples.shape[1] > 0  # Draws
    assert slope_samples.shape[2] == 3  # Features
    assert slope_samples.shape[3] == 4  # Harmonics (including baseline)


def test_extract_predictions(viet_idata_path, viet_model_input):
    """Test extracting prediction samples from last year"""
    idata = az.from_netcdf(viet_idata_path)

    # Create FourierParametrization instance
    model = FourierParametrization(FourierHyperparameters(n_harmonics=3))

    # Extract predictions
    predictions = model.extract_predictions(idata.posterior, viet_model_input)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions dims: {predictions.dims}")
    print(f"Last month: {viet_model_input.last_month}")
    print(f"Prediction months: {predictions.coords['month'].values}")

    # Check shape
    expected_n_pred_months = 12 - (viet_model_input.last_month + 1)
    assert predictions.dims == ('chain', 'draw', 'location', 'month')
    assert predictions.shape[0] == 4  # chains
    assert predictions.shape[1] == 500  # draws
    assert predictions.shape[2] == 19  # locations (Vietnam dataset)
    assert predictions.shape[3] == expected_n_pred_months  # prediction months

    # Check coordinates exist and are labeled
    assert 'location' in predictions.coords
    assert 'month' in predictions.coords
    assert len(predictions.coords['location']) == 19

    # Compute mean prediction per location
    pred_mean = predictions.mean(dim=['chain', 'draw'])
    print(f"\nMean predictions shape: {pred_mean.shape}")
    print(f"Sample predictions for first location:\n{pred_mean.isel(location=0).values}")


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
        posterior = pm.sample_posterior_predictive(idata, var_names=['y_obs', 'A']).posterior_predictive


    # Extract posterior predictions
    mu_posterior = posterior['mu']  # (chain, draw, location, year, month)
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
