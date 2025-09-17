import cyclopts
import numpy as np
import pandas as pd
import pymc as pm

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
    posterior_samples = np.expm1(posterior_samples)
    for l_id, location in enumerate(locations):
        for t_id, time_period in enumerate(new_time_periods):
            samples = posterior_samples[l_id, t_id, -n_samples:]
            new_row = [location, time_period] + samples.tolist()
            rows.append(new_row)

    return pd.DataFrame(rows, columns=colnames)


class SeasonalRegression:
    features = ['mean_temperature']
    def __init__(self, prediction_length=3, lag=3, mcmc_params=MCMCParams()):
        self._prediction_length = prediction_length
        self._lag = lag
        self._mcmc_params = mcmc_params

    def predict(self, training_data: pd.DataFrame) -> pd.DataFrame:
        training_data['y'] = np.log1p(training_data['disease_cases']).interpolate()
        seasonal_data = SeasonalTransform(training_data)
        y = seasonal_data['y']
        y = y[:, 1:]
        X = {feature: seasonal_data[feature] for feature in self.features}

        L, Y, M = y.shape  # Locations, Years, Months
        mean_y = np.nanmean(y, axis=-1, keepdims=True) # L, Y, 1
        max_y = np.nanmax(y, axis=-1, keepdims=True) # L, Y, 1
        std_y = np.nanstd(y, axis=-1, keepdims=True) # L, Y, 1
        base = (y - mean_y) #/ np.maximum(std_y, 0.001)  # L, Y, M
        sample_std = np.nanstd(base, axis=1, keepdims=True)  # L, 1, M
        sample_mean = np.nanmean(base, axis=1, keepdims=True)
        last_month = seasonal_data.last_seasonal_month
        temp = X['mean_temperature'][:, 1:, last_month-self._lag+1:last_month+1]
        n_outcomes = 1  # mean only
        with pm.Model() as model:
            #Regression
            alpha = pm.Normal('intercept', mu=0, sigma=10, shape=(L, 1, n_outcomes))
            beta = pm.Normal('slope', mu=0, sigma=10, shape=(self._lag, n_outcomes))
            sigma = pm.HalfNormal('sigma', sigma=1)
            eta = pm.Deterministic('eta',
                                   alpha + (temp[..., :self._lag] @ beta[:self._lag]))
            sampled_eta = pm.Normal('sampled_eta', mu=eta, sigma=sigma, shape=(L, Y, n_outcomes))
            mu = sampled_eta[..., [0]]
            #std = np.exp(sampled_eta[..., [1]])

            samples = pm.Normal('samples', mu=sample_mean, sigma=sample_std, shape=(L, Y, M))
            transformed_samples = pm.Deterministic('transformed_samples', samples + mu)
            valid_slice = slice(0, -1, 1)
            pm.Normal('observed', mu=transformed_samples[:, valid_slice], sigma=0.1, observed=y[:, valid_slice])
            pm.Normal('y_obs', mu=transformed_samples[:, -1:, :last_month + 1], sigma=0.1, observed=y[:, -1:, :last_month + 1])
            idata = pm.sample(**self._mcmc_params.model_dump())

        posterior_samples = idata.posterior['transformed_samples'].stack(samples=("chain", "draw")).values[:, -1, last_month+1:last_month+self._prediction_length+1]
        preds = np.expm1(posterior_samples)
        mask = preds < 0
        return create_output(training_data, preds)

def test_seasonal_regression(df: pd.DataFrame):
    model = SeasonalRegression(mcmc_params=MCMCParams().debug())
    preds = model.predict(df)
    assert preds.shape == (7, 3, 20), preds

def main(csv_file: str):
    df = pd.read_csv(csv_file)
    model = SeasonalRegression()
    preds = model.predict(df)
    preds.to_csv('seasonal_regression_output.csv', index=False)

if __name__ == '__main__':
    cyclopts.run(main)
