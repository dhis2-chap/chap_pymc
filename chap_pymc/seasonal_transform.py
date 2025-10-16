from typing import Literal

#import altair
import numpy as np
import pandas as pd
import logging

import pydantic
import xarray

logger = logging.getLogger(__name__)


class TransformParameters(pydantic.BaseModel):
    min_prev_months: int | None = None
    min_post_months: int | None = None
    alignment: Literal['min', 'max', 'med'] = 'min'


class SeasonalTransform:
    '''
    This class is responsible for converting time sereies data into a seasonal format. (i.e n_locations, n_seasons, n_months) array
    . The year starts at the month with the lowest average incidence
    of disease cases.
    '''
    def coords(self):
        return {
            'location': self._df['location'].unique(),
            'year': np.arange(self._df['season_idx'].nunique()-1),
            'month': np.arange(self.first_seasonal_month, self.first_seasonal_month + 12 + self._pad_left + self._pad_right),
        }


    def __init__(self, df: pd.DataFrame, params: TransformParameters = TransformParameters()):
        '''
        df: DataFrame with columns ['location', 'time_period', target_name]
        target_name: Name of the target variable column in df
        pad_left: Number of months to pad on the left (before the first month)
        pad_right: Number of months to pad on the right (after the last month)
        0 padding means no padding.
        '''
        min_prev_months = params.min_prev_months
        min_post_months = params.min_post_months
        self._params = params
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
        self._pad_left = max(0, min_prev_months - self.last_seasonal_month - 1) if min_prev_months is not None else 0
        logger.info(f"min_prev_months: {min_prev_months} last seasonal month: {self.last_seasonal_month}, pad_left: {self._pad_left}")
        self.last_seasonal_month+=self._pad_left
        self.first_seasonal_month+=self._pad_left
        self._pad_right = max(self.last_seasonal_month+min_post_months-12+1, 0) if min_post_months is not None else 0
        self._remove_first_year = self.first_seasonal_month > 0

    def _find_min_month(self):
        means = [(month, group['y'].mean()) for month, group in self._df.groupby('month')]
        min_month, val  = min(means, key=lambda x: x[1])
        max_month, val = max(means, key=lambda x: x[1])
        print(f"min_month: {min_month}, max_month: {max_month}")


        med = (min_month+max_month-6)/2
        med = int(med-1) % 12 + 1
        if self._params.alignment == 'min':
            return min_month
        else:
            return med


    def get_df(self, feature_name, start_year=None):
        array = self[feature_name]
        rows = [
            {
                'location': loc,
                'season_idx': season_idx,
                'seasonal_month': month_idx,
                feature_name: array[loc_idx, season_idx, month_idx]
            }
            for loc_idx, loc in enumerate(self._df['location'].unique())
            for season_idx in range(start_year, array.shape[1])
            for month_idx in range(array.shape[2])
        ]
        return pd.DataFrame(rows)


    def plot_feature(self, feature_name):
        import altair as alt
        df = self.get_df(feature_name, start_year=1)
        chart = alt.Chart(df).mark_line().encode(
            x='seasonal_month',
            y=feature_name,
            color='season_idx:O'
        ).facet(
            row=alt.Facet('season_idx:O', title='Season Index'),
            column=alt.Facet('location:N', title='Location')
        ).properties(
            title=f'Seasonal plot of {feature_name} (min month={self._min_month})'
        )
        return chart

    def get_xarray(self, feature_name) -> xarray.DataArray:
        s = self._df.set_index(['location', 'season_idx', 'seasonal_month'])[feature_name].sort_index()
        return s.to_xarray()

    def __getitem__(self, feature_name) -> np.ndarray:
        locations = self._df['location'].unique()
        n_locations = len(locations)
        n_seasons = self._df['season_idx'].nunique()
        n_months = 12
        data_array = np.full((n_locations, n_seasons, n_months), np.nan)
        location_to_idx = {loc: idx for idx, loc in enumerate(locations)}
        if self._pad_right:
            pad_array = np.full((n_locations, n_seasons, self._pad_right), np.nan)
        if self._pad_left:
            left_pad_array = np.full((n_locations, n_seasons, self._pad_left), np.nan)

        for _, row in self._df.iterrows():
            loc_idx = location_to_idx[row['location']]
            season_idx = row['season_idx']
            month_idx = row['seasonal_month']
            data_array[loc_idx, season_idx, month_idx] = row[feature_name]
            if self._pad_right and month_idx < self._pad_right and season_idx > 0:
                pad_array[loc_idx, season_idx-1, month_idx] = row[feature_name]
            if self._pad_left and month_idx+self._pad_left>=n_months and season_idx < n_seasons-1:
                left_pad_array[loc_idx, season_idx+1, month_idx+self._pad_left-n_months] = row[feature_name]
        if self._pad_right:
            logger.info(f"Padding {self._pad_right} months to the right")
            data_array = np.concatenate([data_array, pad_array], axis=-1)
        if self._pad_left:
            logger.info(f"Padding {self._pad_left} months to the left")
            data_array = np.concatenate([left_pad_array, data_array], axis=-1)
        return data_array[:, 1:]


def test_seasonal_transform(df: pd.DataFrame):
    df['y'] = np.log1p(df['disease_cases'])
    st = SeasonalTransform(df)
    pivoted = st['y']
    assert pivoted.shape == (7, 13, 12), pivoted

def test_xarray(colombia_df: pd.DataFrame):
    df = colombia_df
    df['y'] = np.log1p(df['disease_cases'])
    y = SeasonalTransform(df).get_xarray('y')
    mean_y = y.mean(dim='seasonal_month')
    temp = SeasonalTransform(df).get_xarray('mean_temperature')
    corr = xarray.corr(mean_y, temp, dim='season_idx')
    df = corr.to_dataframe(name='correlation').reset_index()
    chart = altair.Chart(df).mark_bar().encode(
        x='seasonal_month:O',
        y='correlation:Q',
        color=altair.Color('correlation:Q', scale=altair.Scale(scheme='redblue', domain=[-1, 1]))
    ).facet(
        facet='location:N',
        columns=4
    ).properties(
        title='Correlation between mean seasonal disease cases and mean temperature by seasonal month and location'
    )
    chart.save('seasonal_correlation.html')
    chart.save('seasonal_correlation.png')

def test_right_pad(df: pd.DataFrame):
    df['y'] = np.log1p(df['disease_cases'])
    st = SeasonalTransform(df, TransformParameters(min_post_months=7))
    pivoted = st['y']
    assert pivoted.shape == (7, 13, 13), pivoted

def test_left_pad(df: pd.DataFrame):
    df['y'] = np.log1p(df['disease_cases'])
    st = SeasonalTransform(df, TransformParameters(min_prev_months=7))
    pivoted = st['y']
    assert pivoted.shape == (7, 13, 13), pivoted