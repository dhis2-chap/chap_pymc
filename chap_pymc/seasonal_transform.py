import numpy as np
import pandas as pd


class SeasonalTransform:
    '''
    This class is responsible for converting time sereies data into a seasonal format. (i.e n_locations, n_seasons, n_months) array
    . The year starts at the month with the lowest average incidence
    of disease cases.
    '''
    def __init__(self, df: pd.DataFrame, target_name='disease_cases'):
        self._df = df.copy()
        self._df['month'] = self._df['time_period'].apply(lambda x: int(x.split('-')[1]))
        self._df['year'] = self._df['time_period'].apply(lambda x: int(x.split('-')[0]))
        self._min_month = self._find_min_month()
        self._df['seasonal_month'] = (self._df['month'] - self._min_month) % 12
        offset = (self._df['month'] - self._min_month) // 12
        self._df['season_idx'] = self._df['year'] + offset
        self._df['season_idx'] = self._df['season_idx'] - self._df['season_idx'].min()
        total_month = self._df['season_idx'] * 12 + self._df['seasonal_month']
        self.first_seasonal_month = (total_month.min()) % 12
        self.last_seasonal_month = (total_month.max()) % 12

    def _find_min_month(self):
        means = ((month, group['y'].mean()) for month, group in self._df.groupby('month'))
        min_month, val  = min(means, key=lambda x: x[1])
        return min_month

    def __getitem__(self, feature_name) -> np.ndarray:
        locations = self._df['location'].unique()
        n_locations = len(locations)
        n_seasons = self._df['season_idx'].nunique()
        n_months = 12
        data_array = np.full((n_locations, n_seasons, n_months), np.nan)
        location_to_idx = {loc: idx for idx, loc in enumerate(locations)}
        for _, row in self._df.iterrows():
            loc_idx = location_to_idx[row['location']]
            season_idx = row['season_idx']
            month_idx = row['seasonal_month']
            data_array[loc_idx, season_idx, month_idx] = row[feature_name]
        return data_array




def test_seasonal_transform(df: pd.DataFrame):
    df['y'] = np.log1p(df['disease_cases'])
    st = SeasonalTransform(df)
    pivoted = st['y']
    assert pivoted.shape == (7, 13, 12), pivoted
