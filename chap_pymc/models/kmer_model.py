from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray
from matplotlib import pyplot as plt
import pymc as pm
import pymc.dims as pmd
from pydantic import BaseModel
from scipy.stats import poisson
import arviz as az
np.random.seed(42)
class HMCParams(BaseModel):
    chains: int = 4
    draws: int = 500
    tune: int = 500

class KmerModel:
    def __init__(self, idata=None, n_counts: int = None, params: HMCParams = HMCParams()):
        self.posterior = idata
        self.n_counts = n_counts
        self.params = params

    def fit(self, df):
        counts = self.create_training_array(df)
        self.fit_counts(counts)

    def fit_counts(self, counts):
        counts.sel(label_positive=False).plot.hist()
        plt.show()
        coords = {key: val.values for key, val in counts.coords.items()}
        self.n_counts = int(counts.sum())
        with pm.Model(coords=coords) as model:
            self.get_model(counts)
            idata = pm.sample(**self.params.model_dump())
            az.plot_posterior(idata, var_names=['h_log_rate', 'sigma'])
            plt.show()
        self.posterior = idata.posterior

    def create_training_array(self, df) -> xarray.DataArray:
        array: xarray.DataArray = (df.set_index(['label_positive', 'kmer'])["count"].sort_index()).to_xarray()
        array = array.fillna(0)
        counts = array
        return counts

    def create_subset_training_array(self, df, metadata, n_repertoirs=None) -> xarray.DataArray:
        counts: xarray.DataArray = (df.set_index(['kmer', 'repertoire_id'])[["count"]].sort_index()).to_xarray()
        counts = counts.fillna(0)
        if n_repertoirs is not None:
            counts = counts.isel(repertoire_id=slice(0, n_repertoirs))
        return ds.sum(dim='repertoire_id')


    def save(self, filename: Path):
        self.posterior.to_netcdf(filename)
        with open(filename.with_suffix('.txt'), "w") as f:
            f.write(str(int(self.n_counts)))

    @classmethod
    def load(cls, filename: Path):
        idata = xarray.open_dataset(filename)
        counts = int(open(filename.with_suffix('.txt')).read())
        return cls(idata, counts)


    def predict(self, r_df: pd.DataFrame):
        r_array = r_df.set_index(['kmer', 'repertoire_id'])["count"].sort_index().to_xarray()
        r_array = r_array.fillna(0)
        labels = {}
        scores = []
        pos_rates = self.posterior['pos_rates'].values
        neg_rates = self.posterior['neg_rates'].values
        plt.scatter(neg_rates, pos_rates)
        plt.title('Negative rates vs. Positive rates')
        plt.show()
        for r_id in r_array.coords['repertoire_id'].values:

            array = r_array.sel(repertoire_id=r_id).values.astype(int)
            ratio = float(array.sum() / self.n_counts)
            p_pos = poisson.logpmf(array, pos_rates*ratio)

            p_neg = poisson.logpmf(array, neg_rates*ratio)
            diff = p_pos.sum() - p_neg.sum()
            scores.append(diff)
            labels[r_id] = float(diff)

        plt.hist(scores, bins=100)
        plt.title('Scores hist')
        plt.show()
        return labels

    def get_model(self, counts: xarray.DataArray):
        mu = float(np.log(counts.sum()/2/len(counts.coords['kmer'].values)))
        print('exp', mu)
        h_log_rate = pm.Normal('h_log_rate') * 5 + mu
        sigma = pm.HalfNormal('sigma', 5)
        log_rates = pmd.Normal(
            'log_rates',
            0,
            1, dims=('kmer', )) * sigma + h_log_rate
        log_diff = pmd.Normal('log_diff', 0, 1, dims=('kmer', ))
        neg_rates = pm.Deterministic('neg_rates', np.exp(log_rates.values), dims=('kmer', ))
        pos_rates = pm.Deterministic('pos_rates', np.exp((log_diff + log_rates).values), dims=('kmer', ))
        pm.Poisson('obs_neg', neg_rates, observed=counts.sel(label_positive=False).values, dims=('kmer', ))
        pm.Poisson('obs_pos', pos_rates, observed=counts.sel(label_positive=True).values, dims=('kmer', ))




@pytest.fixture
def df()->pd.DataFrame:
    return pd.read_csv("/Users/knutdr/Sources/predict-airr/tests/test_output/kmers.csv")

@pytest.fixture
def r_df()->pd.DataFrame:
    return pd.read_csv("/Users/knutdr/Sources/predict-airr/tests/test_output/reportoire_counts.csv")

@pytest.fixture
def metadata()->pd.DataFrame:
    return pd.read_csv("/Users/knutdr/Sources/predict-airr/test_data/train_datasets/train_dataset_1/metadata.csv")

@pytest.fixture
def real_metadata():
    return pd.read_csv("/Users/knutdr/Data/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets/train_dataset_1/metadata.csv")

@pytest.fixture
def real_df():
    return pd.read_csv("/Users/knutdr/Sources/predict-airr/tests/test_output/real_kmers.csv")

@pytest.fixture
def real_r_df():
    return pd.read_csv("/Users/knutdr/Sources/predict-airr/tests/test_output/real_reportoire_counts.csv")

def test_kmers(df, r_df, metadata, name='tmp', n_repertoires=None):
    model = KmerModel()#params=HMCParams(tune=20, draws=20))
    #counts = model.create_training_array(df)
    counts = model.create_subset_training_array(r_df, n_repertoires)
    model.fit_counts(counts)
    model.save(Path(f'{name}.nc'))
    labels = model.predict(r_df)
    for _, row in metadata.iterrows():
        print(row['label_positive'], labels[row['repertoire_id']])

    print(labels)

def test_real(real_df, real_r_df, real_metadata):
    test_kmers(real_df, real_r_df, real_metadata, name='real', n_repertoires=5)

def test_predict_real(real_r_df, real_metadata):
    test_predict(real_r_df, real_metadata)

def test_predict(r_df, metadata, name='tmp'):
    model = KmerModel.load(Path(f'{name}.nc'))
    labels = model.predict(r_df)
    true_scores = []
    false_scores = []
    for _, row in metadata.iterrows():
        print(row['label_positive'], labels[row['repertoire_id']])
        if row['label_positive']:
            true_scores.append(labels[row['repertoire_id']])
        else:
            false_scores.append(labels[row['repertoire_id']])
    plt.hist(true_scores, bins=100, alpha=0.5)
    plt.hist(false_scores, bins=100, alpha=0.5)
    plt.show()

