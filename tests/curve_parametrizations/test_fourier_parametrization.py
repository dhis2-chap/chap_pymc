import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray

from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
    FourierParametrization,
)
from chap_pymc.curve_parametrizations.fourier_parametrization_plots import (
    plot_vietnam_faceted_predictions,
)
from chap_pymc.inference_params import InferenceParams
from chap_pymc.transformations.model_input_creator import ModelInputCreator
from chap_pymc.models.seasonal_fourier_regression import SeasonalFourierRegression, SeasonalFourierRegressionV2


@pytest.fixture()
def coords():
    return {
        'location': np.arange(3),
        'epi_year': np.arange(4),
        'epi_offset': np.arange(12)
    }


@pytest.fixture
def y(coords):
    '''
    DataArray With dimensions (location, epi_year, epi_offset)
    locations: 3
    years: 4 (epi years)
    months: 12
    '''
    L, Y, M = len(coords['location']), len(coords['epi_year']), len(coords['epi_offset'])
    months = np.arange(M)
    locs = np.arange(L)
    years = np.arange(Y)
    pattern = np.sin(2 * np.pi * months / 12)
    y = locs[:, None, None] + years[None, :, None] + pattern[None, None, :]
    y = xarray.DataArray(y, dims=['location', 'epi_year', 'epi_offset'])
    return y


@pytest.fixture()
def vietnam_ds(viet_begin_season):
    creator = ModelInputCreator(prediction_length=3, lag=3)
    model_input = creator.create_model_input(viet_begin_season)
    return model_input

@pytest.fixture()
def viet_ds(viet_begin_season):
    creator = ModelInputCreator(prediction_length=3, lag=3)
    ds, mapping = creator.v2(viet_begin_season, viet_begin_season)
    return creator.to_xarray(model_input)

@pytest.fixture
def vietnam_y_xarray(vietnam_ds):
    """Get y-xarray for Vietnam dataset"""
    return vietnam_ds.y


def test_vietnam_y_xarray_fixture(vietnam_y_xarray):
    """Test that the Vietnam y-xarray fixture works correctly"""
    # Check that it's an xarray DataArray
    assert isinstance(vietnam_y_xarray, xarray.DataArray)

    # Check that it has the expected dimensions
    assert vietnam_y_xarray.dims == ('location', 'epi_year', 'epi_offset')

    # Check that it has coordinates
    assert 'location' in vietnam_y_xarray.coords
    assert 'epi_year' in vietnam_y_xarray.coords
    assert 'epi_offset' in vietnam_y_xarray.coords

    # Check shape makes sense
    assert vietnam_y_xarray.shape[2] == 12  # 12 months

    print(f"Vietnam y-xarray shape: {vietnam_y_xarray.shape}")
    print(f"Dimensions: {vietnam_y_xarray.dims}")
    print(f"Locations: {list(vietnam_y_xarray.coords['location'].values)}")
    print(f"Years: {list(vietnam_y_xarray.coords['epi_year'].values)}")


def test_fourier_parametrization(y, coords):
    """Test Fourier parametrization with synthetic data and create faceted plot"""
    from chap_pymc.curve_parametrizations.fourier_parametrization_plots import (
        plot_faceted_predictions,
    )
    coords |= {'harmonic': np.arange(0, 4)}  # Add harmonic coordinate (0=baseline, 1-3=harmonics for n_harmonics=3)
    with pm.Model(coords=coords):
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
def viet_coords(vietnam_ds):
    n_harmonics = 3  # Number of oscillating harmonics (not counting baseline)

    return {
        'location': vietnam_ds.y.coords['location'].values,
        'year': vietnam_ds.y.coords['year'].values,
        'month': vietnam_ds.y.coords['month'].values,
        'harmonic': np.arange(0, n_harmonics + 1),  # [0, 1, 2, 3]: 0=baseline, 1-3=oscillating
        'feature': vietnam_ds.X.coords['feature'].values
    }
@pytest.fixture
def viet_idata_path():
    return 'vietnam_fourier_parametrization_fit.nc'

@pytest.fixture()
def nepal_model_input(nepal_data):
    creator = ModelInputCreator(prediction_length=3, lag=3, mask_empty_seasons=False)
    model_input = creator.create_model_input(nepal_data)
    return creator.to_xarray(model_input)

@pytest.mark.slow
def test_nepal_regresion(nepal_data):
    # Split data into training and future
    training_data = nepal_data.iloc[:-3]
    future_data = nepal_data.iloc[-3:]

    # Use V2 implementation with recent improvements
    model = SeasonalFourierRegressionV2(
        params=SeasonalFourierRegressionV2.Params(
            fourier_hyperparameters=FourierHyperparameters(
                n_harmonics=2,
                use_prev_year=True  # Test the new prev_year feature
            ),
            inference_params=InferenceParams(
                method='hmc',
                draws=500,
                tune=500,
                progressbar=True
            )
        )
    )

    # Get predictions
    preds_df = model.predict(training_data, future_data)

    # Get raw samples for visualization
    ds, mapping = model.get_input_data(future_data, training_data)
    samples = model.get_raw_samples(ds)

    # Check for NaNs
    assert not samples.isnull().any(), "Model produced NaN predictions for Nepal data"

    # Calculate statistics for plotting
    mu_mean = samples.mean(dim='samples')
    mu_lower = samples.quantile(0.025, dim='samples')
    mu_upper = samples.quantile(0.975, dim='samples')

    # Create plot using plotting function
    plot_vietnam_faceted_predictions(ds['y'], mu_mean, mu_lower, mu_upper,
                                    ds.coords, output_file='nepal_fourier_fit.png')

@pytest.mark.slow
def test_full_vietnam_regression(viet_full_year, skip=7):
    for i, (viet_instance, future) in enumerate(viet_full_year):
        if i<skip:
            continue
        creator = ModelInputCreator(prediction_length=3, lag=3)
        ds, mapping = creator.v2(viet_instance, future)
        test_vietnam_regression(ds, viet_idata_path=f'viet_{i}.nc', i=i)

def test_full_nepal_regression(nepal_full_year):
    test_full_vietnam_regression(nepal_full_year, skip=0)


@pytest.fixture()
def vietnam_ds(viet_full_year) -> xarray.Dataset:
    viet_instance, future = next(viet_full_year)
    creator = ModelInputCreator(prediction_length=3, lag=3)
    ds, mapping = creator.v2(viet_instance, future)
    return ds

@pytest.mark.slow
def test_vietnam_regression(vietnam_ds, viet_idata_path=None, i=0):
    ds = vietnam_ds
    m = FourierParametrization(FourierHyperparameters(n_harmonics=3))
    viet_coords = {dim: ds[dim].values for dim in ds.dims} | m.extra_dims
    #viet_coords= vietnam_ds.coords() | {'harmonic': np.arange(0, 4)}  # Add harmonic coordinate (0=baseline, 1-3=harmonics)

    with pm.Model(coords=viet_coords) as model:
        m.get_regression_model(vietnam_ds.X, vietnam_ds.y)
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
        vietnam_ds.y.to_netcdf(f'y_{viet_idata_path}')
        vietnam_ds.X.to_netcdf(f'X_{viet_idata_path}')

    #posterior = idata.posterior
    #posterior['A'].isel(harmonic=0).median(dim=['chain', 'draw']).plot()
    # Extract posterior predictions
    mu_posterior = posterior['y_obs']  # (chain, draw, location, year, month)
    mu_mean = mu_posterior.mean(dim=['chain', 'draw'])  # (location, year, month)
    mu_lower = mu_posterior.quantile(0.025, dim=['chain', 'draw'])
    mu_upper = mu_posterior.quantile(0.975, dim=['chain', 'draw'])

    # Create plot using plotting function
    plot_vietnam_faceted_predictions(vietnam_ds.y, mu_mean, mu_lower, mu_upper, viet_coords,
                                     output_file=f'vietnam_fourier_fit_{i}.png')

def test_vietnam(viet_begin_season, debug_model):
    debug_model.predict(viet_begin_season)


def test_nepal(nepal_data: pd.DataFrame, debug_model):
    global TESTING
    TESTING = True
    debug_model.predict(nepal_data)

@pytest.fixture
def debug_model() -> SeasonalFourierRegression:
    return SeasonalFourierRegression(
        prediction_length=3,
        lag=3,
        fourier_hyperparameters=FourierHyperparameters(n_harmonics=2, do_mixture=True),
        inference_params=InferenceParams(method='advi', n_iterations=1_000, progressbar=True)
    )

@pytest.mark.skip(reason="Requires viet_idata_path fixture")
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


@pytest.mark.skip(reason="Requires viet_idata_path fixture")
def test_extract_predictions(viet_idata_path, vietnam_ds):
    """Test extracting prediction samples from last year"""
    idata = az.from_netcdf(viet_idata_path)

    # Create FourierParametrization instance
    model = FourierParametrization(FourierHyperparameters(n_harmonics=3))

    # Extract predictions
    predictions = model.extract_predictions(idata.posterior, vietnam_ds)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions dims: {predictions.dims}")
    print(f"Last month: {vietnam_ds.last_month}")
    print(f"Prediction months: {predictions.coords['month'].values}")

    # Check shape
    expected_n_pred_months = 12 - (vietnam_ds.last_month + 1)
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


@pytest.mark.slow
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



@pytest.mark.slow
def test_vietnam_parameter_correlations(vietnam_ds):
    """Fit Fourier model and plot parameter-feature correlations"""
    from chap_pymc.curve_parametrizations.fourier_parametrization_plots import (
        plot_parameter_feature_correlations,
    )

    # Extract coordinates and data
    vietnam_y = vietnam_ds.y
    n_harmonics = 3
    coords = {
        'location': vietnam_y.coords['location'].values,
        'year': vietnam_y.coords['year'].values,
        'month': vietnam_y.coords['month'].values,
        'harmonic': np.arange(0, n_harmonics+1)  # Add harmonic coordinate (0=baseline, 1-n=harmonics)
    }



    # Build and sample the model
    with pm.Model(coords=coords):
        FourierParametrization(FourierHyperparameters(n_harmonics=n_harmonics)).get_model(vietnam_y)
        idata = pm.sample(draws=500, tune=500, progressbar=True, return_inferencedata=True)

    # Create correlation plot
    plot_parameter_feature_correlations(idata, vietnam_ds, n_harmonics,
                                        output_file='vietnam_parameter_correlations.html')
