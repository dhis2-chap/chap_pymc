from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import cyclopts
import pytest
from pandas import DataFrame


def main():
    df = pd.read_csv('/Users/knutdr/Downloads/dataset2/training_data.csv')
    return create_data_arrays(df)


def create_data_arrays(df: pd.DataFrame, horizon=3):
    df['log1p'] = np.log1p(df['disease_cases'])
    month_index = df['time_period'].apply(lambda x: int(x.split('-')[1]))
    df['month'] = month_index
    #Find month with minimum log1p
    means = ((month, group['disease_cases'].mean()) for month, group in df.groupby('month'))
    min_month, val  = min(means, key=lambda x: x[1])

    all_means = []
    all_stds = []
    full_year_data = []
    for location, group in df.groupby('location'):

        group = group.interpolate()
        cutoff_month_index = np.flatnonzero(df['month'] == min_month)[0]
        extra_offset = (len(group)-cutoff_month_index+horizon)%12
        ds = group['log1p'].values
        assert not np.isnan(ds).any(), ds
        assert not np.isinf(ds).any(), ds
        ds = np.append(ds, [np.nan]*horizon)
        normies = []
        year_data_per_loc = []
        extra_offset = extra_offset if extra_offset <= horizon else 0
        for i in range(len(ds) // 12):
            year_data = ds[cutoff_month_index + i * 12:cutoff_month_index + (i+1) * 12+extra_offset]
            missing = 12+extra_offset-len(year_data)
            if missing > 0:
                year_data = np.append(year_data, [np.nan]*missing)
            year_data_per_loc.append(year_data)
            normalized = (year_data - year_data[:12].mean()) / max(year_data[:12].std(), 0.001)
            normies.append(normalized)
            plt.plot(normalized)
        year_data_per_loc = np.array(year_data_per_loc)
        full_year_data.append(year_data_per_loc)
        normies = np.array(normies)
        means = np.nanmean(normies[...], axis=0)
        stds = np.nanstd(normies[...], axis=0)
        assert not np.isnan(means).any(), (normies, stds)
        assert not np.isnan(stds).any(), (normies, stds)
        all_means.append(means)
        all_stds.append(stds)
        h = means + stds
        l = means - stds
        # plt.plot(h, c='k', ls='--')
        # plt.plot(means, c='k', ls='--')
        # plt.plot(l, c='k', ls='--')

        # group = group.reset_index().iloc[:12]
        # plt.plot( group['disease_cases'])
        plt.title(location)
        #plt.show()
    return np.array(all_means), np.array(all_stds), np.array(full_year_data),missing+3


def make_model(all_means: 'L, M', all_stds, full_year_data: 'loc, year, month', missing:int=0):
    L, Y, M = full_year_data.shape
    train_data = full_year_data[:, :-1, :]
    test_data = full_year_data[:, -1:, :]
    repeat_year = lambda a: np.repeat(a, Y, axis=0).reshape(L,Y,M)
    all_means = repeat_year(all_means)
    all_stds = repeat_year(all_stds)
    with pm.Model() as model:
        sigma = pm.HalfNormal('sigma', 3.0)
        mu = pm.Normal('mu', 0.0, 10.)
        yearly_mean = pm.Normal('yearly_mean', mu=mu,sigma=10, shape=(L, Y,1))
        yearly_std = pm.HalfNormal('yearly_std', sigma=sigma, shape=(L, Y, 1))
        normalized = pm.Normal('normalized', mu=all_means, sigma=all_stds)
        pred_means = pm.Deterministic('pred_mean', normalized * yearly_std+yearly_mean)

        pm.Normal('train_obs', mu=pred_means[:, :-1, :], sigma=0.1, observed=train_data)
        seen_months = M-missing
        pm.Normal('test_obs', mu=pred_means[:, -1:, :seen_months], sigma=0.1, observed=test_data[:, :, :seen_months])
        idata = pm.sample(
            draws=500,
            chains=4,
            tune=500,
            progressbar=True)

    return idata

def create_output(training_pdf, posterior_samples, horizon=3, n_samples=100, missing=3):
    locations = training_pdf['location'].unique()
    last_time_idx = training_pdf['time_period'].max()
    year, month = map(int, last_time_idx.split('-'))
    raw_months = np.arange(horizon)+month
    new_months = (raw_months % 12)+1
    new_years = year + raw_months//12
    print(year, raw_months, new_months, new_years)
    new_time_periods = [f'{y:d}-{m:02d}' for y, m in zip(new_years, new_months)]
    colnames = ['location', 'time_period'] + [f'sample_{i}' for i in range(n_samples)]
    rows =  []
    M = posterior_samples.shape[-2]
    posterior_samples = np.expm1(posterior_samples)
    for l_id, location in enumerate(locations):
        for t_id, time_period in enumerate(new_time_periods):
            samples = posterior_samples[l_id, -1, M-missing+t_id, -n_samples:]
            new_row = [location, time_period] + samples.tolist()
            rows.append(new_row)

    return pd.DataFrame(rows, columns=colnames)


def plot_predictions(idata, full_year_data):
    L = len(full_year_data)
    qhigh = idata.posterior['pred_mean'].quantile(dim=('draw', 'chain'), q=0.9).values
    qlow = idata.posterior['pred_mean'].quantile(dim=('draw', 'chain'), q=0.1).values
    md = idata.posterior['pred_mean'].median(dim=('draw', 'chain')).values
    for i in range(L):
        for year in (-1, -2):
            plt.plot(qlow[i, year], ls='--')
            plt.plot(qhigh[i, year], ls='--')
            plt.plot(md[i, year], ls='--')
            plt.plot(full_year_data[i, year])
            plt.title(f'year={year}, i={i}')
            #plt.show()

@pytest.fixture(scope='module')
def posterior_samples():
    return np.random.rand(100*7*12*15).reshape((7, 12, 15, 100))+1

@pytest.fixture(scope='module')
def training_df():
    return pd.read_csv('/Users/knutdr/Downloads/dataset2/training_data.csv')

@pytest.fixture(scope='module')
def training_df2():
    return pd.read_csv(Path(__file__).parent.parent/'test_data'/'training_data.csv')

def test_inhomog_create(training_df2):
    create_data_arrays(training_df2)

def test_output_predictions(training_df, posterior_samples):
    create_output(training_df,posterior_samples)


def test_make_data(training_df):
    all_means, all_stds, full_year_data,missing = create_data_arrays(training_df)
    idata = make_model(all_means, all_stds, full_year_data, missing)
    posterior_samples = idata.posterior['pred_mean'].stack(dim={'samples':('draw', 'chain')}).values
    return create_output(training_df, posterior_samples, missing)

app = cyclopts.App()

@app.command()
def train(train_data: str, model: str, model_config: str, force=False):
    return

@app.command()
def predict(model: str,
            historic_data: str,
            future_data: str,
            out_file: str,
            model_config: str | None = None):
    training_df = pd.read_csv(historic_data)
    predictions = get_predictions(training_df)
    predictions.to_csv(out_file, index=False)


def get_predictions(training_df: DataFrame) -> DataFrame:
    all_means, all_stds, full_year_data, missing = create_data_arrays(training_df)
    idata = make_model(all_means, all_stds, full_year_data,missing=missing)
    posterior_samples = idata.posterior['pred_mean'].stack(dim={'samples': ('draw', 'chain')}).values
    predictions = create_output(training_df, posterior_samples,missing=missing)
    return predictions


if __name__ == '__main__':
    app()