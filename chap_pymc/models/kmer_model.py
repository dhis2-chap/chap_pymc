from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray
from matplotlib import pyplot as plt
import pymc as pm
import pymc.dims as pmd

class KmerModel:
    def __init__(self, df: pd.DataFrame):
        array: xarray.DataArray = (df.set_index(['label_positive', 'kmer'])["count"].sort_index()).to_xarray()
        print(array)
        plt.hist(array.sel(label_positive=True), bins=100, color='r', density=True, alpha=0.5)
        plt.hist(array.sel(label_positive=False), bins=100, color='b', density=True, alpha=0.5)
        plt.legend()
        #plt.show()
        self.counts = array

    def fit(self):
        coords = {key: val.values for key, val in self.counts.coords.items()}
        with pm.Model(coords=coords) as model:
            self.get_model(self.counts)
            idata = pm.sample(tune=400, draws=400, chains=4)
        median_diff = idata.posterior['log_diff'].median(dim=('chain', 'draw'))
        print(median_diff)
        print(np.sort(median_diff.values)[-20:])
        #median_diff.plot.hist(bins=100)
        plt.show()

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

def test_kmers(df):
    model = KmerModel(df)
    model.fit()
    print(df.head())

