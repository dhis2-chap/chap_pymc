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
            plt.plot(normalized, 'o')
        plt.title(str(location)+'norm')
        plt.show()
        year_data_per_loc = np.array(year_data_per_loc)
        full_year_data.append(year_data_per_loc)
        normies = np.array(normies)
        means = np.nanmean(normies[...], axis=0)
        stds = np.nanstd(normies[...], axis=0)
        assert not np.isnan(means).any(), (normies, stds)
        assert not np.isnan(stds).any(), (normies, stds)
        all_means.append(means)
        all_stds.append(stds)
        fully_norm = (normies-means)/stds
        #mark os
        plt.plot(fully_norm)

        plt.show()
        #plt.plot(stds)
        #plt.show()
        h = means + stds
        l = means - stds
        # plt.plot(h, c='k', ls='--')
        # plt.plot(means, c='k', ls='--')
        # plt.plot(l, c='k', ls='--')

        # group = group.reset_index().iloc[:12]
        #plt.plot( group['disease_cases'])
        plt.title(location)
        #plt.show()
    return np.array(all_means), np.array(all_stds), np.array(full_year_data),missing+3

def chatgpted_model(full_year_data, missing:int=3):
    L, Y, M = full_year_data.shape
    seen_months = M - missing



def make_full_model(full_year_data: 'loc, year, month', missing: int=3):
    L, Y, M = full_year_data.shape
    seen_months = M - missing
    with pm.Model() as model:
        a = pm.Normal('a', mu=0, sigma=10, shape=(L, Y, 1))
        b = pm.Normal('b', mu=0, sigma=10, shape=(L, Y, 1))
        c = pm.Normal('c', mu=0, sigma=10, shape=(L, Y, 1))
        t = np.arange(0, M).reshape(1, 1, M)
        #eta = pm.Deterministic('eta', a*t**2+b*t+c)
        all_mu = pm.Deterministic('all_mu', a*t**2+b*t+c)
        if False:
            yearly_offsets = pm.Normal('yearly_offsets', mu=0, sigma=2, shape=(L, Y, 1))
            yearly_scales = pm.HalfNormal('yearly_scales', sigma=2,
                                    shape=(L, Y, 1))
            monthly_shape_raw = pm.Normal('monthly_shape_raw',
                                      mu=0, sigma=2, shape=(L, 1, M))

            monthly_shape = pm.Deterministic('monthly_shape', monthly_shape_raw-monthly_shape_raw.mean(axis=-1, keepdims=True))

        # means are a combination of withinyear monthly shape and offset and scale per year
        # all_mu = pm.Deterministic('all_mu',
        #                           monthly_shape * yearly_scales + yearly_offsets)

        if False:
            location_sigma = pm.HalfNormal('location_sigma', sigma=2,shape=(L, 1, 1))
            month_sigma = pm.HalfNormal('month_sigma', sigma=2,shape=(1, 1, M))
            base_sigma = pm.Deterministic('base_sigma', np.sqrt(location_sigma**2 + month_sigma**2))
        # base_sigma = pm.HalfNormal('base_sigma', sigma=10, shape=(L, 1, M))
        base_sigma = np.full((L, 1, M), 1)
        pm.Normal('seen_years',
                  mu=all_mu[:, :-1, :],
                  sigma=base_sigma,
                  observed=full_year_data[:, :-1, :])
        pm.Normal('seen_same_year',
                  mu=all_mu[:, -1:, :seen_months],
                  sigma=base_sigma[..., :seen_months],
                  observed=full_year_data[:, -1:, :seen_months])
        pm.Normal('pred_mean',
                  mu=all_mu[:, -1:, seen_months:],
                  sigma=base_sigma[..., seen_months:])


        idata = pm.sample(
            draws=500,
            chains=4,
            tune=500,
            progressbar=True)
    return idata


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
        seen_months = M - missing
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
    # idata = make_model(all_means, all_stds, full_year_data, missing)
    idata = make_full_model(full_year_data, missing)
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
    #idata = make_model(all_means, all_stds, full_year_data,missing=missing)
    idata = make_full_model(full_year_data, missing)
    posterior_samples = idata.posterior['pred_mean'].stack(dim={'samples': ('draw', 'chain')}).values
    predictions = create_output(training_df, posterior_samples,missing=missing)
    return predictions


if __name__ == '__main__':
    app()