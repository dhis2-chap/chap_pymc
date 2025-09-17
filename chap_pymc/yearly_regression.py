import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from .season_plot import SeasonCorrelationPlot


def create_data_arrays(seasonal_data: pd.DataFrame, horizon=3):
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
    return X, y, location_idx, n_locations, n_seasons


def basic_season_regression(df: pd.DataFrame):
    """
    Basic PyMC regression model predicting season_max disease cases
    based on mean temperature for each month of the season.
    """
    # Get the seasonal data
    seasonal_data = SeasonCorrelationPlot(df).data()
    X, y, location_idx, n_locations, n_seasons = create_data_arrays(seasonal_data)
    with pm.Model() as model:
        # Priors
        start_period = 3
        alpha = pm.Normal('intercept', mu=0, sigma=10, shape=n_locations)
        beta = pm.Normal('slope', mu=0, sigma=10, shape=(start_period,1))
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value
        mu = alpha[location_idx] + (X[..., :start_period] @ beta[:start_period]).squeeze()

        # Likelihood
        pm.Normal(f'y_obs[{start_period}]', mu=mu, sigma=sigma, observed=y)

        # Sample from posterior
        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    return model, trace

def plot_inference_results(trace):
    """Create inference plots for the regression model."""

    # 1. Trace plots for all intercepts (all locations)
    az.plot_trace(trace, var_names=['intercept'])
    plt.suptitle('Trace Plots: All Location Intercepts')
    plt.savefig('regression_all_intercepts_trace.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Trace plots for all slopes (all seasonal months)
    az.plot_trace(trace, var_names=['slope'])
    plt.suptitle('Trace Plots: All Seasonal Month Slopes')
    plt.savefig('regression_all_slopes_trace.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 3. Posterior summary plot
    az.plot_forest(trace, var_names=['intercept', 'slope'])
    plt.suptitle('Posterior Estimates (95% HDI)')
    plt.savefig('regression_forest.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 4. Seasonal effects plot
    slope_samples = trace.posterior['slope'].values.reshape(-1, 12)
    slope_mean = slope_samples.mean(axis=0)
    slope_hdi = az.hdi(slope_samples, hdi_prob=0.95)

    fig, ax = plt.subplots(figsize=(10, 6))
    months = range(12)
    ax.plot(months, slope_mean, 'o-', color='blue', label='Mean Effect')
    ax.fill_between(months, slope_hdi[:, 0], slope_hdi[:, 1], alpha=0.3, color='blue', label='95% HDI')
    ax.set_xlabel('Seasonal Month')
    ax.set_ylabel('Temperature Effect on Disease Cases')
    ax.set_title('Seasonal Temperature Effects')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('seasonal_effects.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_yearly_max_regression(df):
    seasonal_data = SeasonCorrelationPlot(df).data()
    model, trace = basic_season_regression(df)

    # Print basic summary
    print("Regression Summary:")
    # print(f"Data points used: {len(clean_data)}")
    #print(f"Mean temperature range: {clean_data['mean_temperature'].min():.2f} to {clean_data['mean_temperature'].max():.2f}")
    #print(f"Season max range: {clean_data['season_max'].min():.2f} to {clean_data['season_max'].max():.2f}")

    # Create inference plots
    print("\nGenerating inference plots...")
    plot_inference_results(trace)

    # Print posterior summary
    print("\nPosterior Summary:")
    print(az.summary(trace, var_names=['intercept', 'slope', 'sigma']))

    return model, trace
