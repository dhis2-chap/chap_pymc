import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az



def main():
    df = pd.read_csv('/Users/knutdr/Downloads/dataset2/training_data.csv')
    print(df.head())
    all_means = []
    all_stds = []
    full_year_data = []
    for location, group in df.groupby('location'):
        group = group.interpolate()
        month = np.arange(len(group))%12

        ds = np.log1p(group['disease_cases'].values)
        t = np.bincount(month, weights=ds)
        #plt.plot(t)
        #season_month = 6+np.arange(len(group))

        ds = np.log1p(group['disease_cases'].values)
        normies = []
        year_data_per_loc = []
        for i in range(len(group) // 12):
            year_data = ds[8 + i * 12:20 + i * 12]
            year_data_per_loc.append(year_data)
            normalized = (year_data - year_data.mean()) / year_data.std()
            normies.append(normalized)
            plt.plot(normalized)
        full_year_data.append(year_data_per_loc)
        normies = np.array(normies)
        means = np.nanmean(normies, axis=0)
        stds = np.nanstd(normies, axis=0)
        all_means.append(means)
        all_stds.append(stds)
        h = means+stds
        l = means-stds
        plt.plot(h, c='k', ls='--')
        plt.plot(means, c='k', ls='--')
        plt.plot(l, c='k', ls='--')

        # group = group.reset_index().iloc[:12]
        # plt.plot( group['disease_cases'])
        plt.title(location)
        plt.show()
    return np.array(all_means), np.array(all_stds), np.array(full_year_data)


def make_model(all_means: 'L, M', all_stds, full_year_data: 'loc, year, month'):
    L, Y, M =full_year_data.shape

    repeat_year = lambda a: np.repeat(a, Y, axis=0).reshape(L,Y,M)
    all_means = repeat_year(all_means)
    all_stds = repeat_year(all_stds)
    print(all_means.shape)
    with pm.Model() as model:
        yearly_mean = pm.Normal('yearly_mean', mu=1,sigma=10, shape=(L, Y,1))
        yearly_std = pm.HalfNormal('yearly_std', sigma=10, shape=(L, Y, 1))
        normalized = pm.Normal('normalized', mu=all_means, sigma=all_stds)
        pred_means = pm.Deterministic('pred_mean', normalized*yearly_std+yearly_mean)
        pm.Normal('obs', mu=pred_means, sigma=0.1, observed=full_year_data)
        idata = pm.sample(
            draws=400,
            chains=4,
            tune=400,
            progressbar=True)

    return idata





if __name__ == '__main__':
    all_means, all_stds, full_year_data = main()
    idata = make_model(all_means, all_stds, full_year_data)
    az.plot_posterior(idata)
    plt.show()
