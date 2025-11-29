from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray
from matplotlib import pyplot as plt
import pymc as pm
import pymc.dims as pmd
from scipy.stats import poisson


class KmerModel:
    def __init__(self, idata=None, n_counts: int = None):
        self.posterior = idata
        self.n_counts = n_counts

    def fit(self, df):
        array: xarray.DataArray = (df.set_index(['label_positive', 'kmer'])["count"].sort_index()).to_xarray()
        array = array.fillna(0)
        self.n_counts = int(array.sum())
        counts = array
        coords = {key: val.values for key, val in counts.coords.items()}
        with pm.Model(coords=coords) as model:
            self.get_model(counts)
            idata = pm.sample(tune=400, draws=400, chains=4)
        self.posterior = idata.posterior

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
        for r_id in r_array.coords['repertoire_id'].values:

            array = r_array.sel(repertoire_id=r_id).values.astype(int)
            ratio = float(array.sum()/self.n_counts)
            pos_rates = self.posterior['pos_rates'].values * ratio
            p_pos = poisson.logpmf(array, pos_rates)
            neg_rates = self.posterior['neg_rates'].values*ratio
            p_neg = poisson.logpmf(array, neg_rates)
            diff = p_pos.sum() - p_neg.sum()
            labels[r_id] = bool(p_pos.sum()>p_neg.sum())
        return labels


    def get_model(self, counts):
        h_log_rate = pm.Normal('h_log_rate', 0, 5)
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

def test_kmers(df, r_df):
    model = KmerModel()
    model.fit(df)
    model.save(Path('tmp.nc'))
    labels = model.predict(r_df)
    print(labels)

def test_predict(r_df, metadata):
    model = KmerModel.load(Path('tmp.nc'))
    labels = model.predict(r_df)
    for _, row in metadata.iterrows():
        print(row['label_positive'], labels[row['repertoire_id']])


