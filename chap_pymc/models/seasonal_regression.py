import dataclasses
import logging

import cyclopts
import numpy as np
import pandas as pd
import pymc as pm

from chap_pymc.models.model_with_dimensions import DimensionalModel
from chap_pymc.models.model_with_dimensions import ModelParams as ModelDefParams

try:
    import altair as alt
    import matplotlib.pyplot as plt
    # alt.data_transformers.enable("vegafusion")
except ImportError:
    plt = None
    alt = None
import pytest

from chap_pymc.inference_params import InferenceParams
from chap_pymc.model_input_creator import FullModelInput, ModelInputCreator
from chap_pymc.seasonal_transform import SeasonalTransform

TESTING=False
logging.basicConfig(level=logging.INFO)


def Deterministic(name, value):
    if TESTING:
        return pm.Deterministic(name, value)
    return value


def create_output(training_pdf, posterior_samples, n_samples=1000):
    n_samples = min(n_samples, posterior_samples.shape[-1])
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
    #locs: np.ndarray

class ModelParams(ModelDefParams):
    #errors: Literal['iid', 'rw'] = 'rw'
    mask_empty_seasons: bool = False

class SeasonalRegression:
    def __init__(self, prediction_length=3, lag=3, inference_params=InferenceParams(), model_params=ModelParams()):
        self._prediction_length = prediction_length
        self._lag = lag
        self._inference_params = inference_params
        self._model_params = model_params
        self._explanation_plots = []

    @property
    def explanation_plots(self):
        return self._explanation_plots

    def predict(self, training_data: pd.DataFrame, return_idata=False):
        model_input = self.create_model_input(training_data)
        with pm.Model() as _:
            self.define_stable_model(model_input)
            idata = pm.sample(**self._inference_params.model_dump())
        if TESTING and False:
            self.plot_effect_trace(idata)
            self.pyplot_last_year(model_input, idata)
        last_month = model_input.last_month
        posterior_samples = idata.posterior['transformed_samples'].stack(samples=("chain", "draw")).values[:, -1, last_month+1:last_month+self._prediction_length+1]
        preds = np.expm1(posterior_samples)

        if return_idata:
            return create_output(training_data, preds), idata
        else:
            return create_output(training_data, preds)

    def predict_with_dims(self, training_data: pd.DataFrame, n_samples=1000):
        creator = ModelInputCreator(
            prediction_length=self._prediction_length,
            lag=self._lag,
            mask_empty_seasons=self._model_params.mask_empty_seasons
        )
        model_input = creator.create_model_input(training_data)
        model_input = creator.to_xarray(model_input)
        self._seasonal_data = creator.seasonal_data

        coords = creator.seasonal_data.coords() | {'feature': [f'temp_lag{self._lag-i}'for i in range(self._lag)]}
        with pm.Model(coords=coords):
            DimensionalModel(self._model_params).build_model(model_input)
            # define_stable_model(model_input, self._model_params)
            #graph = pm.model_to_graphviz(model)
            #graph.render('model_graph', format='png', view=True)
            approx = pm.fit(n=self._inference_params.n_iterations, method='advi')
        last_month = model_input.last_month
        posterior_samples = approx.sample(n_samples)
        #az.plot_posterior(posterior_samples, var_names=['slope', 'intercept', 'scale_mu', 'scale_sigma'])
        #plt.show()
        posterior_samples = posterior_samples.posterior['transformed_samples'].stack(samples=("chain", "draw")).values[
            :, -1, last_month + 1:last_month + self._prediction_length + 1]
        preds = np.expm1(posterior_samples)
        return create_output(training_data, preds)


    def predict_advi(self, training_data: pd.DataFrame, n_samples=1000, return_approx=False):
        model_input = self.create_model_input(training_data)

        with pm.Model():
            self.define_stable_model(model_input)
            approx = pm.fit(n=self._inference_params.n_iterations, method='advi')

        # Draw samples from the approximation
        posterior_samples = approx.sample(n_samples)
        print(posterior_samples, type(posterior_samples))
        # Extract transformed_samples predictions
        last_month = model_input.last_month
        #transformed_samples = posterior_samples.posterior['transformed_samples']  # shape: (n_samples, L, Y, M)
        #pred_samples = transformed_samples[:, :, -1, last_month+1:last_month+self._prediction_length+1]  # (n_samples, L, prediction_length)
        posterior_samples = posterior_samples.posterior['transformed_samples'].stack(samples=("chain", "draw")).values[
            :, -1, last_month + 1:last_month + self._prediction_length + 1]
        #preds = np.expm1(posterior_samples)
        # Transpose to match expected format (L, prediction_length, n_samples)
        #pred_samples = np.transpose(pred_samples, (1, 2, 0))
        preds = np.expm1(posterior_samples)

        if return_approx:
            return create_output(training_data, preds), approx
        else:
            return create_output(training_data, preds)

    def create_model_input(self, training_data: pd.DataFrame) -> FullModelInput:
        creator = ModelInputCreator(
            prediction_length=self._prediction_length,
            lag=self._lag,
            mask_empty_seasons=self._model_params.mask_empty_seasons
        )
        model_input = creator.create_model_input(training_data)
        self._seasonal_data = creator.seasonal_data
        return model_input

    def plot_model_input(self, model_input: FullModelInput):
        L, Y, M = model_input.y.shape
        for _loc in range(L):
            ...

    def define_model(self, model_input: FullModelInput) -> int:
        L, Y, M = model_input.y.shape
        n_outcomes = 1  # mean only
        alpha = pm.Normal('intercept', mu=0, sigma=10, shape=(L, 1, n_outcomes))  # SHould this be global?
        beta = pm.Normal('slope', mu=0, sigma=10, shape=(self._lag, n_outcomes))

        # sigma = pm.HalfNormal('sigma', sigma=10, shape=(L, 1, 1))

        eta = alpha + (model_input.X[..., :self._lag] @ beta[:self._lag])
        eta = Deterministic('eta', eta)


        scale = pm.LogNormal('scale', sigma=100, shape=(L, Y, 1))

        # Maybe clearer to just add epsilon noise here
        epsilon = pm.Normal('epsilon', mu=0, sigma=40, shape=(L, Y, n_outcomes))
        sampled_eta = Deterministic('sampled_eta', eta + epsilon)
        # sampled_eta = pm.Normal('sampled_eta', mu=eta, sigma=sigma, shape=(L, Y, n_outcomes))

        mu = sampled_eta[..., [0]]
        Deterministic('transformed_pattern', model_input.seasonal_pattern * scale + mu)  # For plotting

        samples = pm.Normal('samples',
                            mu=model_input.seasonal_pattern,
                            sigma=model_input.seasonal_errors,
                            shape=(L, Y, M))

        transformed_samples = pm.Deterministic('transformed_samples', samples * scale + mu)
        valid_slice = slice(0, -1, 1)
        last_year_slice = slice(None, model_input.last_month + 1)
        pm.Normal('observed', mu=transformed_samples[:, valid_slice], sigma=0.1, observed=model_input.y[:, valid_slice])
        pm.Normal('y_obs',
                  mu=transformed_samples[:, -1:, last_year_slice],
                  sigma=0.1,
                  observed=model_input.y[:, -1:, last_year_slice])
        return Y

    def plot_effect_trace(self, idata):
        beta = idata.posterior['slope'].stack(samples=("chain", "draw")).values
        print(beta.shape)
        for lag in range(self._lag):
            plt.hist(beta[lag, 0, ...], density=True, bins=100)
            plt.title(f"{lag}")
            plt.show()

    def define_stable_model(self, model_input: FullModelInput):
        L, Y, M = model_input.y.shape
        if model_input.X.size:
            X = model_input.X
            alpha = pm.Normal('intercept', mu=0, sigma=10, shape=(L, 1, 1))  # SHould this be global?
            beta = pm.Normal('slope', mu=0, sigma=10, shape=(X.shape[-1], 1))
            eta = Deterministic('eta',
                                   alpha + (X @ beta))
        else:
            eta = 0


        loc_mu = pm.Normal('loc_mu', mu=0, sigma=10, shape=(L, 1, 1))
        loc_sigma = pm.HalfNormal('loc_sigma',sigma=10)
        loc = pm.Normal('loc', mu=loc_mu, sigma=loc_sigma, shape=(L, Y, 1)) + eta
        scale_mu = pm.Normal('scale_mu', mu=1, sigma=1, shape=(L, 1, 1))
        scale_sigma = pm.HalfNormal('scale_sigma', sigma=1)
        scale = pm.Normal('scale', scale_mu, sigma=scale_sigma, shape=(L, Y, 1))

        transformed_pattern = Deterministic('transformed_pattern',
                model_input.seasonal_pattern * scale+loc)

        if self._model_params.errors == 'rw':
            ar_sigma = pm.HalfNormal('ar_sigma', sigma=0.2, shape=(L, 1))
            init_dist = pm.Normal.dist(np.zeros((L, Y)), ar_sigma)
            epsilon = pm.GaussianRandomWalk('epsilon',
                                            init_dist=init_dist,
                                            mu=0,
                                            sigma=ar_sigma,
                                            steps=M-1,
                                            shape=(L, Y, M))
        else:
            epsilon = 0
        transformed_samples = pm.Deterministic('transformed_samples', transformed_pattern+epsilon)
        sigma = pm.HalfNormal('sigma', sigma=1, shape=(L, 1, 1))
        pm.Normal('y_obs', mu=transformed_samples[:, :-1],
                  sigma=sigma,
                  observed=model_input.y[:, :-1])

        pm.Normal('last_year',
                  mu=transformed_samples[:, -1:, :model_input.last_month+1],
                  sigma=sigma,
                  observed=model_input.y[:, -1:, :model_input.last_month+1])



    def _get_longform_trace(self, idata, param_names: list[str]):
        ...

    def set_explanation_plots(self, training_data: FullModelInput, idata, year_idx=None):
        param_names = ['eta', 'sampled_eta', 'samples', 'transformed_samples', 'transformed_pattern', 'epsilon', 'scale']
        median_dict = {name: idata.posterior[name].median(dim=['chain', 'draw']).values for name in param_names}

        L, Y, M = training_data.y.shape  # Locations, Years, Months
        plot_year_idx = Y - 1 if year_idx is None else year_idx
        # Create data in long format for Altair
        data = []

        for loc in range(L):
            location_name = f'Location {loc}'

            # Get median values for this location and specified year
            median_dict['eta'][loc, plot_year_idx, 0]
            sampled_eta_val = median_dict['sampled_eta'][loc, plot_year_idx, 0]
            scale_val = median_dict['scale'][loc, plot_year_idx, 0]
            mu_val = sampled_eta_val  # mu is sampled_eta for the last dimension

            # Update location name to include median scale and mu
            location_name = f'Location {loc} (scale={scale_val:.3f}, μ={mu_val:.3f})'

            # Get arrays for this location/year
            transformed_samples = median_dict['transformed_samples'][loc, plot_year_idx, :]
            transformed_pattern = median_dict['transformed_pattern'][loc, plot_year_idx, :]
            epsilon = median_dict['epsilon'][loc, plot_year_idx, :]
            y_obs = training_data.y[loc, plot_year_idx, :]
            training_data.seasonal_pattern[loc, 0, :] + sampled_eta_val
            transformed_minus_epsilon = transformed_pattern - epsilon

            # Add monthly varying variables
            for month in range(M):
                seasonal_month = month + 1
                data.extend([
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': transformed_samples[month], 'variable': 'transformed_samples',
                     'line_type': 'solid'},
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': transformed_pattern[month], 'variable': 'transformed_pattern',
                     'line_type': 'solid'},
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': transformed_minus_epsilon[month], 'variable': 'transformed_samples - epsilon',
                     'line_type': 'solid'},
                    # {'location': location_name, 'seasonal_month': seasonal_month,
                    #  'value': seasonal_pattern_plus_sampled_eta[month], 'variable': 'seasonal_pattern + sampled_eta',
                    #  'line_type': 'solid'},
                    {'location': location_name, 'seasonal_month': seasonal_month,
                     'value': y_obs[month], 'variable': 'y_obs',
                     'line_type': 'solid'},
                    # {'location': location_name, 'seasonal_month': seasonal_month,
                    #  'value': eta_val, 'variable': 'eta',
                    #  'line_type': 'dashed'},
                    # {'location': location_name, 'seasonal_month': seasonal_month,
                    #  'value': sampled_eta_val, 'variable': 'sampled_eta',
                    #  'line_type': 'dotted'}
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
            title=f'Median Parameter Values (Year {plot_year_idx})'
        )
        #self._explanation_plots.append(chart)
        #chart.save('seasonal_regression_explanation.html')
        self._explanation_plots.append(chart)

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

    def pyplot_last_year(self, model_input, idata):
        pre_noise = idata.posterior['transformed_pattern'].median(dim=('chain', 'draw')).values
        pred_line = idata.posterior['transformed_samples']
        med = pred_line.median(dim=('chain', 'draw')).values
        upper = pred_line.quantile(0.75, dim=('chain', 'draw')).values
        lower = pred_line.quantile(0.25, dim=('chain', 'draw')).values
        L = med.shape[0]
        for location in range(L):
            pred = med[location, -1]
            u = upper[location, -1]
            l = lower[location, -1]
            plt.plot(pred, label='med', color='blue')
            plt.plot(u, label='upper', color= 'blue')
            plt.plot(l, label='lower', color='blue')
            plt.plot(pre_noise[location, -1], label=f'Prenoise {location+1}', color='green')
            plt.plot(model_input.seasonal_pattern[location,0])
            plt.plot(model_input.y[location, -1], color='red')
            plt.legend()
            plt.show()


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

def test_nepal(nepal_data: pd.DataFrame):
    global TESTING
    TESTING = True
    model = SeasonalRegression(inference_params=InferenceParams(chains=2, tune=200, draws=200),
                               model_params=ModelParams(errors='rw', use_mixture=True),
                               lag=3, prediction_length=3)

    model.predict_with_dims(nepal_data)
    #for plot in model.explanation_plots:
    #    plot.show()

def test_seasonal_regression(large_df: pd.DataFrame):
    global TESTING
    # TESTING = True
    model = SeasonalRegression(inference_params=InferenceParams(chains=4, tune=400, draws=400).debug(),
                               model_params=ModelParams(errors='rw'),
                               lag=3, prediction_length=3)

    preds, idata = model.predict(large_df, return_idata=True)
    model.plot_prediction(idata, large_df, 'prediction_plot.png')
    for plot in model.explanation_plots:
        plot.show()


def test_advi(large_df: pd.DataFrame):
    global TESTING
    #TESTING = True
    model = SeasonalRegression(inference_params=InferenceParams().debug(),
                               model_params=ModelParams(errors='rw'),
                               lag=3, prediction_length=3)

    model.predict_with_dims(large_df, n_samples=100)
    #model.plot_prediction(approx, large_df, 'advi_prediction_plot.png')
    #for plot in model.explanation_plots:
    #    plot.show()

def test_begin_season(thai_begin_season):
    #first_group = next(iter(thai_begin_season.groupby('location')))[1]
    #hai_begin_season = first_group
    model = SeasonalRegression(inference_params=InferenceParams(), lag=3, prediction_length=3)
    model.predict(thai_begin_season)

def test_viet_begin_season(viet_begin_season):
    global TESTING
    TESTING = True
    model = SeasonalRegression(inference_params=InferenceParams(), lag=3, prediction_length=3)
    model.predict_with_dims(viet_begin_season)

@pytest.fixture()
def model_input():
    L, Y, M = 1, 2, 12
    months = np.array(L * [[np.arange(M)]])
    return FullModelInput(
        X=np.random.rand(L, Y, 3),
        y=np.sin(2 * np.pi * months // 12) * np.array([[1], [-1]]),
        seasonal_pattern=np.sin(2 * np.pi * months // 12),
        seasonal_errors=np.ones((L, 1, M)),
        last_month=3
    )

def test_anti_pattern(model_input):
    params = InferenceParams(chains=2, draws=200, tune=500)
    reg = SeasonalRegression(inference_params=params, lag=3, prediction_length=3)
    with pm.Model():
        reg.define_stable_model(model_input)
        idata = pm.sample(**params.model_dump())
    posterior = idata.posterior
    scale = posterior['scale'].median(dim=['chain', 'draw'])
    all_scales = posterior['scale'].stack(samples=("chain", "draw")).values
    tp = posterior['transformed_samples'].median(dim=['chain', 'draw'])[0]
    for i in (0, 1):
        plt.plot(tp[i])
        plt.plot(model_input.y[0, i])
        plt.show()
    for y in all_scales[0]:
        plt.hist(y[0], bins=50, density=True)
    plt.show()
    #reg.set_explanation_plots(model_input, idata, Y-1)
    #for i, plot in enumerate(reg.explanation_plots):
    #    plot.save(f'anti_pattern_explanation_{i}.html')
    scales = scale.values.ravel()
    assert scales[-1] < 0.1, scale.values
    assert scales[0] > 0.5, scale.values

def test_altair():
    chart = alt.Chart(pd.DataFrame({
        'x': np.arange(10),
        'y': np.random.rand(10),
        'category': ['A'] * 5 + ['B'] * 5
    })).mark_line().encode(
        x='x',
        y='y',
        color='category'
    )
    chart.save('test_altair.png')

def test_sample_broadcasting():
    means = np.arange(6).reshape((2, 1, 3))
    print(means)
    samples = pm.draw(pm.Normal.dist(mu=means, sigma=0.1, shape=(2, 4, 3)))
    print(samples)

app = cyclopts.App()

@app.command()
def predict(csv_file: str):
    df = pd.read_csv(csv_file)
    model = SeasonalRegression()
    if False:
        preds, idata = model.predict(df, return_idata=True)
    else:
        preds = model.predict_advi(df, return_approx=False, n_samples=1000)
    # save data from idata
    #idata.to_netcdf('seasonal_regression_trace.nc')
    #model.plot_trace(idata, 'seasonal_regression_trace.png')

    #model.plot_prediction(idata, df, 'seasonal_regression_predictions.png')
    preds.to_csv('seasonal_regression_output.csv', index=False)

@app.command()
def explain(csv_file: str, output_folder: str = '.', inference_params: InferenceParams = InferenceParams()):
    import chap_core
    from chap_core.assessment.dataset_splitting import train_test_generator
    dataset = chap_core.data.DataSet.from_csv(csv_file)
    train_data, test_instances  = train_test_generator(dataset, prediction_length=3, n_test_sets=12)
    for t, (historic, _future, _truth) in enumerate(test_instances):
        df = historic.to_pandas()
        df.time_period = df.time_period.astype(str)
        model = SeasonalRegression(inference_params=inference_params)
        model.predict(df)
        assert len(model.explanation_plots)>0
        for i, plot in enumerate(model.explanation_plots):
            plot.save(f'{output_folder}/seasonal_regression_explanation_{t}_{i}.html')



def test_explain(data_path, tmp_path):
    explain(data_path/'training_data.csv', tmp_path, InferenceParams().debug())



if __name__ == '__main__':
    app()
    #app()
