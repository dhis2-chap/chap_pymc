import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .season_plot import SeasonCorrelationPlot


def create_data_arrays(seasonal_data: pd.DataFrame, horizon=3):
    # Remove any rows with missing values
    outcome = 'std'
    clean_data = seasonal_data.dropna(subset=[f'season_{outcome}', 'mean_temperature'])
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
            y.append(data[f'season_{outcome}'].values[0])
            location_idx.append(i)
    X = np.array(X)
    X = (X - X.mean())/X.std()  # Centering the predictor
    y = np.array(y)
    location_idx = np.array(location_idx)
    return X, y, location_idx, n_locations, n_seasons


def basic_season_regression(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Basic PyMC regression model predicting season_max disease cases
    based on mean temperature for each month of the season.
    """
    # Get the seasonal data
    seasonal_data = SeasonCorrelationPlot(df).data()
    X, y, location_idx, n_locations, n_seasons = create_data_arrays(seasonal_data)

    # Split data into train and test
    X_train, X_test, y_train, y_test, loc_idx_train, loc_idx_test = train_test_split(
        X, y, location_idx, test_size=test_size, random_state=random_state, stratify=location_idx
    )
    do_location_beta = False
    with (pm.Model() as model):
        start_period = 6
        alpha = pm.Normal('intercept', mu=0, sigma=10, shape=n_locations)
        beta = pm.Normal('slope', mu=0, sigma=10, shape=(start_period,1))
        sigma = pm.HalfNormal('sigma', sigma=1)
        # Expected value for training data
        mu_train = pm.Deterministic('mu_train', alpha[loc_idx_train] + (X_train[..., :start_period] @ beta[:start_period]).squeeze())
        # Likelihood
        pm.Normal(f'y_obs[{start_period}]', mu=mu_train, sigma=sigma, observed=y_train)

        # Sample from posterior
        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    return model, trace, (X_train, X_test, y_train, y_test, loc_idx_train, loc_idx_test)

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

def evaluate_model_performance(trace, X_train, X_test, y_train, y_test, loc_idx_train, loc_idx_test):
    """Evaluate model performance on train and test sets."""

    # Extract posterior samples
    alpha_samples = trace.posterior['intercept'].values.reshape(-1, trace.posterior['intercept'].shape[-1])
    beta_samples = trace.posterior['slope'].values.reshape(-1, trace.posterior['slope'].shape[-2])
    mu_samples = trace.posterior['mu_train'].values.reshape(-1, trace.posterior['mu_train'].shape[-1])
    start_period = beta_samples.shape[1]

    def predict_samples(X, loc_idx):
        """Generate prediction samples for given data."""
        n_samples = alpha_samples.shape[0]
        n_obs = X.shape[0]
        predictions = np.zeros((n_samples, n_obs))

        for i in range(n_samples):
            mu = alpha_samples[i][loc_idx] + (X[..., :start_period] @ beta_samples[i].reshape(-1, 1)).squeeze()
            predictions[i] = mu

        return predictions

    # Generate predictions
    train_pred_samples = predict_samples(X_train, loc_idx_train)
    test_pred_samples = predict_samples(X_test, loc_idx_test)

    # Point predictions (posterior mean)
    y_train_pred = train_pred_samples.mean(axis=0)
    y_test_pred = test_pred_samples.mean(axis=0)

    # Calculate metrics
    train_metrics = {
        'mse': mean_squared_error(y_train, y_train_pred),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred)
    }

    test_metrics = {
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred)
    }

    # Print results
    print("\n=== Model Performance ===")
    print("Training Set:")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  R²: {train_metrics['r2']:.4f}")

    print("\nTest Set:")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")

    # Prediction vs actual plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training set
    ax1.scatter(y_train, y_train_pred, alpha=0.6, color='blue')
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'Training Set (R² = {train_metrics["r2"]:.3f})')
    ax1.grid(True, alpha=0.3)

    # Test set
    ax2.scatter(y_test, y_test_pred, alpha=0.6, color='green')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Test Set (R² = {test_metrics["r2"]:.3f})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
    plt.show()

    return train_metrics, test_metrics, (y_train_pred, y_test_pred)

def test_yearly_max_regression(df):
    seasonal_data = SeasonCorrelationPlot(df).data()
    model, trace, (X_train, X_test, y_train, y_test, loc_idx_train, loc_idx_test) = basic_season_regression(df)

    # Print basic summary
    print("Regression Summary:")
    print(f"Training data points: {len(y_train)}")
    print(f"Test data points: {len(y_test)}")
    print(f"Total data points: {len(y_train) + len(y_test)}")

    # Create inference plots
    print("\nGenerating inference plots...")
    plot_inference_results(trace)

    # Evaluate model performance
    train_metrics, test_metrics, predictions = evaluate_model_performance(
        trace, X_train, X_test, y_train, y_test, loc_idx_train, loc_idx_test
    )

    # Print posterior summary
    print("\nPosterior Summary:")
    print(az.summary(trace, var_names=['intercept', 'slope', 'sigma']))

    return model, trace, train_metrics, test_metrics
