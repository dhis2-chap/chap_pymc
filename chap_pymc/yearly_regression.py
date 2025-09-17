import pymc as pm
import numpy as np
import pandas as pd
from .season_plot import SeasonCorrelationPlot

def basic_season_regression(df: pd.DataFrame):
    """
    Basic PyMC regression model predicting season_max disease cases
    based on mean temperature for each month of the season.
    """
    # Get the seasonal data
    seasonal_data = SeasonCorrelationPlot(df).data()

    # Remove any rows with missing values
    clean_data = seasonal_data.dropna(subset=['season_max', 'mean_temperature'])

    # Extract predictor and response variables
    X = []
    y = []
    location_idx = []
    n_locations = len(clean_data['location'].unique())
    n_seasons = 12
    for season_idx in clean_data['season_idx'].unique():
        for i, location in enumerate(clean_data['location'].unique()):
            data = clean_data[(clean_data['season_idx'] == season_idx) & (clean_data['location'] == location)]

            if not len(data) == n_seasons:
                continue
            seasonal_month = data['seasonal_month'].values
            mean_temp = data['mean_temperature'].values
            a = np.zeros(n_seasons)
            a[seasonal_month] = mean_temp
            X.append(a)
            y.append(data['season_max'].values[0])
            location_idx.append(i)
    X = np.array(X)
    X = (X - X.mean())/X.std()  # Centering the predictor
    y = np.array(y)
    location_idx = np.array(location_idx)

    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('intercept', mu=0, sigma=10, shape=n_locations)
        beta = pm.Normal('slope', mu=0, sigma=10, shape=(n_seasons,1))
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value
        mu = alpha[location_idx] + (X @ beta).squeeze()

        # Likelihood
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

        # Sample from posterior
        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    return model, trace, clean_data

def test_yearly_max_regression(df):
    seasonal_data = SeasonCorrelationPlot(df).data()
    model, trace, clean_data = basic_season_regression(df)

    # Print basic summary
    print("Regression Summary:")
    print(f"Data points used: {len(clean_data)}")
    print(f"Mean temperature range: {clean_data['mean_temperature'].min():.2f} to {clean_data['mean_temperature'].max():.2f}")
    print(f"Season max range: {clean_data['season_max'].min():.2f} to {clean_data['season_max'].max():.2f}")

    return model, trace, clean_data
