import dataclasses
from statistics import median

import cyclopts
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import altair as alt

from chap_pymc.mcmc_params import MCMCParams
from chap_pymc.seasonal_transform import SeasonalTransform


def create_output(training_pdf, posterior_samples, n_samples=100):
    horizon = posterior_samples.shape[-2]
    locations = training_pdf['location'].unique()
    last_time_idx = training_pdf['time_period'].max()
    year, month = map(int, last_time_idx.split('-'))
    raw_months = np.arange(horizon) + month
    new_months = (raw_months % 12) + 1
    new_years = year + raw_months // 12
    print(year, raw_months, new_months, new_years)
    new_time_periods = [f'{y:d}-{m:02d}' for y, m in zip(new_years, new_months)]
    colnames = ['location', 'time_period'] + [f'sample_{i}' for i in range(n_samples)]
    rows = []

    #M = posterior_samples.shape[-2]
    # posterior_samples = np.expm1(posterior_samples)
    for l_id, location in enumerate(locations):
        for t_id, time_period in enumerate(new_time_periods):
            samples = posterior_samples[l_id, t_id, -n_samples:]
            new_row = [location, time_period] + samples.tolist()
            rows.append(new_row)

    return pd.DataFrame(rows, columns=colnames)

@dataclasses.dataclass
class TrainingArrays:
    X: np.ndarray  # (locations, seasons, months, lag)
    y: np.ndarray  # (locations, seasons, months)
    seasonal_pattern: np.ndarray
    locs: np.ndarray

class SeasonalRegression:
    features = ['mean_temperature']
    def __init__(self, prediction_length=3, lag=3, mcmc_params=MCMCParams()):
        self._prediction_length = prediction_length
        self._lag = lag
        self._mcmc_params = mcmc_params

    def predict(self, training_data: pd.DataFrame, return_idata=False):
        training_data['y'] = np.log1p(training_data['disease_cases']).interpolate()
        seasonal_data = SeasonalTransform(training_data, min_prev_months=self._lag, min_post_months=self._prediction_length)
        y = seasonal_data['y']
        y = y[:, 1:]
        X = {feature: seasonal_data[feature] for feature in self.features}

        L, Y, M = y.shape  # Locations, Years, Months
        mean_y = np.nanmean(y, axis=-1, keepdims=True) # L, Y, 1
        base = (y - mean_y) #/ np.maximum(std_y, 0.001)  # L, Y, M
        std_per_mont_per_loc = np.nanstd(base, axis=1, keepdims=True)  # L, 1, M
        seasonal_pattern = np.nanmean(base, axis=1, keepdims=True)
        self._seasonal_pattern = seasonal_pattern
        last_month = seasonal_data.last_seasonal_month
        temp = X['mean_temperature'][:, 1:, last_month-self._lag+1:last_month+1]
        temp = (temp - np.nanmean(temp)) / np.nanstd(temp)  # Standardize predictor
        n_outcomes = 1  # mean only
        with pm.Model() as model:
            #Regression
            alpha = pm.Normal('intercept', mu=0, sigma=10, shape=(L, 1, n_outcomes))
            beta = pm.Normal('slope', mu=0, sigma=10, shape=(self._lag, n_outcomes))

            sigma = pm.HalfNormal('sigma', sigma=5)
            eta = pm.Deterministic('eta',
                                   alpha + (temp[..., :self._lag] @ beta[:self._lag]))
            scale = pm.HalfNormal('scale', sigma=5, shape=(L, Y, 1))
            # Maybe clearer to just add epsilon noise here
            sampled_eta = pm.Normal('sampled_eta', mu=eta, sigma=sigma, shape=(L, Y, n_outcomes))


            mu = sampled_eta[..., [0]]

            samples = pm.Normal('samples',
                                mu=seasonal_pattern,
                                sigma=std_per_mont_per_loc,
                                shape=(L, Y, M)) # Maybe shape is wrong should be (L, M, Y).reshape(L, Y, M)

            transformed_samples = pm.Deterministic('transformed_samples', samples*scale + mu)
            valid_slice = slice(0, -1, 1)

            pm.Normal('observed', mu=transformed_samples[:, valid_slice], sigma=0.1, observed=y[:, valid_slice])
            pm.Normal('y_obs',
                      mu=transformed_samples[:, -1:, :last_month + 1],
                      sigma=0.1,
                      observed=y[:, -1:, :last_month + 1])

            idata = pm.sample(**self._mcmc_params.model_dump())

        posterior_samples = idata.posterior['transformed_samples'].stack(samples=("chain", "draw")).values[:, -1, last_month+1:last_month+self._prediction_length+1]
        preds = np.expm1(posterior_samples)
        self.set_explanation_plots(
            TrainingArrays(X=temp, y=y, seasonal_pattern=seasonal_pattern, locs=mean_y),
            idata)

        if return_idata:
            return create_output(training_data, preds), idata
        else:
            return create_output(training_data, preds)

    def set_explanation_plots(self, training_data: TrainingArrays, idata):
        param_names = ['eta', 'sampled_eta', 'samples', 'transformed_samples']
        median_dict = {name: idata.posterior[name].median(dim=['chain', 'draw']).values for name in param_names}

        L, Y, M = training_data.y.shape  # Locations, Years, Months
        last_year_idx = Y - 1

        # Create data in long format for Altair
        data = []

        for loc in range(L):
            location_name = f'Location {loc}'

            # Get median values for this location and last year
            eta_val = median_dict['eta'][loc, last_year_idx, 0]
            sampled_eta_val = median_dict['sampled_eta'][loc, last_year_idx, 0]
            epsilon = sampled_eta_val - eta_val

            # Get arrays for this location/year
            transformed_samples = median_dict['transformed_samples'][loc, last_year_idx, :]
            y_obs = training_data.y[loc, last_year_idx, :]
            seasonal_pattern_plus_sampled_eta = self._seasonal_pattern[loc, 0, :] + sampled_eta_val
            transformed_minus_epsilon = transformed_samples - epsilon

            # Add monthly varying variables
            for month in range(M):
                seasonal_month = month + 1
                data.extend([
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': transformed_samples[month], 'variable': 'transformed_samples',
                     'line_type': 'solid'},
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': transformed_minus_epsilon[month], 'variable': 'transformed_samples - epsilon',
                     'line_type': 'solid'},
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': seasonal_pattern_plus_sampled_eta[month], 'variable': 'seasonal_pattern + sampled_eta',
                     'line_type': 'solid'},
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': y_obs[month], 'variable': 'y_obs',
                     'line_type': 'solid'},
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': eta_val, 'variable': 'eta',
                     'line_type': 'dashed'},
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': sampled_eta_val, 'variable': 'sampled_eta',
                     'line_type': 'dotted'}
                ])

        df = pd.DataFrame(data)

        # Create chart without interactive selection to avoid duplication issues
        base = alt.Chart(df)

        # Create separate layers for different line types
        solid_lines = base.transform_filter(
            alt.datum.line_type == 'solid'
        ).mark_line(point=True).encode(
            x=alt.X('seasonal_month:O', title='Seasonal Month'),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color('variable:N', title='Variable')
        )

        dashed_lines = base.transform_filter(
            alt.datum.line_type == 'dashed'
        ).mark_line(strokeDash=[5, 5]).encode(
            x='seasonal_month:O',
            y='value:Q',
            color='variable:N'
        )

        dotted_lines = base.transform_filter(
            alt.datum.line_type == 'dotted'
        ).mark_line(strokeDash=[2, 2]).encode(
            x='seasonal_month:O',
            y='value:Q',
            color='variable:N'
        )

        # Combine layers and facet
        chart = (solid_lines + dashed_lines + dotted_lines).facet(
            column=alt.Column('location:N', title=None)
        ).resolve_scale(
            color='independent'
        ).properties(
            title='Median Parameter Values (Last Year)'
        )

        chart.save('seasonal_regression_explanation.html')
        chart.show()

    def plot_trace(self, idata, output_file=None):
        """Plot trace plots for all alpha (intercept) and beta (slope) parameters."""

        # Get parameter data
        alpha_data = idata.posterior['intercept']  # Shape: (chain, draw, L, 1, n_outcomes)
        beta_data = idata.posterior['slope']       # Shape: (chain, draw, lag, n_outcomes)

        n_locations = alpha_data.shape[2]
        n_lags = beta_data.shape[2]

        # Create subplots: one row for each location's alpha, plus one row for all betas
        fig, axes = plt.subplots(n_locations + n_lags, 2, figsize=(12, 3 * (n_locations + n_lags)))

        # Plot alpha (intercept) traces for each location
        for loc in range(n_locations):
            row_idx = loc

            # Extract alpha values for this location
            alpha_values = alpha_data.values[:, :, loc, 0, 0]  # (chain, draw)

            # Trace plot
            ax_trace = axes[row_idx, 0]
            for chain in range(alpha_values.shape[0]):
                ax_trace.plot(alpha_values[chain, :], alpha=0.8, label=f'Chain {chain+1}')
            ax_trace.set_title(f'Alpha[{loc}] - Trace Plot')
            ax_trace.set_xlabel('Draw')
            ax_trace.set_ylabel('Value')
            ax_trace.legend()
            ax_trace.grid(True, alpha=0.3)

            # Posterior distribution
            ax_posterior = axes[row_idx, 1]
            alpha_flat = alpha_values.flatten()
            ax_posterior.hist(alpha_flat, bins=50, density=True, alpha=0.6, edgecolor='black')
            ax_posterior.axvline(np.mean(alpha_flat), color='red', linestyle='-', label='Mean')
            ax_posterior.axvline(np.median(alpha_flat), color='black', linestyle='--', label='Median')
            ax_posterior.set_title(f'Alpha[{loc}] - Posterior Distribution')
            ax_posterior.set_xlabel('Value')
            ax_posterior.set_ylabel('Density')
            ax_posterior.legend()
            ax_posterior.grid(True, alpha=0.3)

        # Plot beta (slope) traces for each lag
        for lag in range(n_lags):
            row_idx = n_locations + lag

            # Extract beta values for this lag
            beta_values = beta_data.values[:, :, lag, 0]  # (chain, draw)

            # Trace plot
            ax_trace = axes[row_idx, 0]
            for chain in range(beta_values.shape[0]):
                ax_trace.plot(beta_values[chain, :], alpha=0.8, label=f'Chain {chain+1}')
            ax_trace.set_title(f'Beta[{lag}] - Trace Plot')
            ax_trace.set_xlabel('Draw')
            ax_trace.set_ylabel('Value')
            ax_trace.legend()
            ax_trace.grid(True, alpha=0.3)

            # Posterior distribution
            ax_posterior = axes[row_idx, 1]
            beta_flat = beta_values.flatten()
            ax_posterior.hist(beta_flat, bins=50, density=True, alpha=0.6, edgecolor='black')
            ax_posterior.axvline(np.mean(beta_flat), color='red', linestyle='-', label='Mean')
            ax_posterior.axvline(np.median(beta_flat), color='black', linestyle='--', label='Median')
            ax_posterior.set_title(f'Beta[{lag}] - Posterior Distribution')
            ax_posterior.set_xlabel('Value')
            ax_posterior.set_ylabel('Density')
            ax_posterior.legend()
            ax_posterior.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Trace plots saved to: {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_prediction(self, idata, training_data, output_file=None):
        """Plot median transformed_samples vs actual observed y, faceted by year (rows) and location (columns)."""

        # Prepare the training data
        training_data_copy = training_data.copy()
        training_data_copy['y'] = np.log1p(training_data_copy['disease_cases']).interpolate()
        seasonal_data = SeasonalTransform(training_data_copy)
        y_observed = seasonal_data['y'][:, 2:]  # Match the slicing in predict method

        # Get the transformed samples from posterior
        transformed_samples = idata.posterior['transformed_samples']  # (chain, draw, locations, years, months)

        # Calculate median across chains and draws
        median_predictions = transformed_samples.median(dim=['chain', 'draw']).values  # (locations, years, months)

        L, Y, M = y_observed.shape
        locations = training_data['location'].unique()

        # Create time indices for plotting
        # Get the starting year from the seasonal transform
        first_year = training_data['time_period'].apply(lambda x: int(x.split('-')[0])).min()
        years = list(range(first_year + 2, first_year + 2 + Y))  # +2 because we slice [:, 2:]
        months = list(range(1, M + 1))

        # Set up the plot - rows for years, columns for locations
        n_locs = len(locations)
        n_years = len(years)

        fig, axes = plt.subplots(n_years, n_locs, figsize=(4 * n_locs, 3 * n_years))

        # Handle different subplot configurations
        if n_years == 1 and n_locs == 1:
            axes = [[axes]]
        elif n_years == 1:
            axes = [axes]
        elif n_locs == 1:
            axes = [[ax] for ax in axes]

        month_labels = [f'M{m}' for m in months]
        x_vals = np.arange(len(month_labels))

        for year_idx, year in enumerate(years):
            for loc_idx, location in enumerate(locations):
                ax = axes[year_idx][loc_idx]

                # Get observed and predicted values for this year-location combination
                y_obs = y_observed[loc_idx, year_idx, :]
                y_pred = median_predictions[loc_idx, year_idx, :]

                # Plot observed vs predicted (only two lines per subplot)
                ax.plot(x_vals, y_obs, 'o-', color='blue', alpha=0.8, linewidth=2,
                       markersize=6, label='Observed')
                ax.plot(x_vals, y_pred, 's--', color='red', alpha=0.8, linewidth=2,
                       markersize=6, label='Predicted')

                # Formatting
                ax.set_title(f'{location} - {year}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Set month labels
                ax.set_xticks(x_vals)
                ax.set_xticklabels(month_labels, rotation=45)

                # Labels only on edge subplots
                if year_idx == n_years - 1:  # Bottom row
                    ax.set_xlabel('Month')
                if loc_idx == 0:  # Left column
                    ax.set_ylabel('Log(Disease Cases + 1)')

                # Legend only on top-right subplot
                if year_idx == 0 and loc_idx == n_locs - 1:
                    ax.legend(fontsize=8)

        # Add overall title
        fig.suptitle('Model Predictions vs Observed Data by Year and Location',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Prediction plot saved to: {output_file}")
        else:
            plt.show()

        plt.close()

def test_seasonal_regression(df: pd.DataFrame):
    model = SeasonalRegression(mcmc_params=MCMCParams().debug(), lag=3, prediction_length=3)
    preds = model.predict(df)

def test_sample_broadcasting():
    means = np.arange(6).reshape((2, 1, 3))
    print(means)
    samples = pm.draw(pm.Normal.dist(mu=means, sigma=0.1, shape=(2, 4, 3)))
    print(samples)
    assert False

def main(csv_file: str):
    df = pd.read_csv(csv_file)
    model = SeasonalRegression()
    preds, idata = model.predict(df, return_idata=True)
    # save data from idata
    idata.to_netcdf('seasonal_regression_trace.nc')
    model.plot_trace(idata, 'seasonal_regression_trace.png')

    model.plot_prediction(idata, df, 'seasonal_regression_predictions.png')
    preds.to_csv('seasonal_regression_output.csv', index=False)


if __name__ == '__main__':
    cyclopts.run(main)
    #app()
