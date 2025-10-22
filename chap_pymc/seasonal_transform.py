from typing import Literal

#import altair
import numpy as np
import pandas as pd
import logging

import pydantic
import xarray

logger = logging.getLogger(__name__)

# Constants
MONTHS_PER_YEAR = 12


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
            'month': np.arange(MONTHS_PER_YEAR + self._pad_left + self._pad_right),
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
        self._df['seasonal_month'] = (self._df['month'] - self._min_month) % MONTHS_PER_YEAR
        offset = (self._df['month'] - self._min_month) // MONTHS_PER_YEAR
        self._df['season_idx'] = self._df['year'] + offset
        self._df['season_idx'] = self._df['season_idx'] - self._df['season_idx'].min()
        total_month = self._df['season_idx'] * MONTHS_PER_YEAR + self._df['seasonal_month']
        # Store raw (unpadded) month indices
        self._first_seasonal_month_raw = (total_month.min()) % MONTHS_PER_YEAR
        self._last_seasonal_month_raw = (total_month.max()) % MONTHS_PER_YEAR

        # Calculate left padding requirement
        # We need enough historical months for lag features
        # Example: if data ends at month 3 and we need min_prev_months=5 lag features,
        # we have months [0,1,2,3] available, so we need 5-3-1=1 month of padding
        # Formula: pad_left = max(0, min_prev_months - last_month - 1)
        self._pad_left = max(0, min_prev_months - self._last_seasonal_month_raw - 1) if min_prev_months is not None else 0
        logger.info(f"min_prev_months: {min_prev_months} last seasonal month: {self._last_seasonal_month_raw}, pad_left: {self._pad_left}")

        # Effective (padded) month indices - used by downstream code
        # These represent where months appear after left padding is applied
        self.last_seasonal_month = self._last_seasonal_month_raw + self._pad_left
        self.first_seasonal_month = self._first_seasonal_month_raw + self._pad_left

        # Calculate right padding requirement
        # If predictions extend beyond month 11 (end of disease year), we need extra months
        # Example: last_month=10 (after left padding), min_post_months=3 predictions
        # Predictions would be months 11, 12, 13 → need 13-12+1=2 months of right padding
        # Formula: pad_right = max(0, last_month + min_post_months - MONTHS_PER_YEAR + 1)
        self._pad_right = max(self.last_seasonal_month + min_post_months - MONTHS_PER_YEAR + 1, 0) if min_post_months is not None else 0
        self._remove_first_year = self.first_seasonal_month > 0

    def _find_min_month(self):
        means = [(month, group['y'].mean()) for month, group in self._df.groupby('month')]
        min_month, val  = min(means, key=lambda x: x[1])
        max_month, val = max(means, key=lambda x: x[1])
        print(f"min_month: {min_month}, max_month: {max_month}")


        med = (min_month+max_month-6)/2
        med = int(med-1) % MONTHS_PER_YEAR + 1
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

    def get_xarray(self, feature_name, drop_first_year=False, add_last_year=False) -> xarray.DataArray:
        s = self._df.set_index(['location', 'season_idx', 'seasonal_month'])[feature_name].sort_index()
        data_array = s.to_xarray()
        if add_last_year:
            # Add an extra year by copying the last year
            last_year_idx = data_array.coords['season_idx'].values.max()
            last_year_data = data_array.sel(season_idx=last_year_idx)
            new_year_idx = last_year_idx + 1
            last_year_data = last_year_data.expand_dims({'season_idx': [new_year_idx]})
            data_array = xarray.concat([data_array, last_year_data], dim='season_idx')

        # Add actual month names as coordinates
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal_months = data_array.coords['seasonal_month'].values

        # Calculate actual calendar months from seasonal months
        # seasonal_month i corresponds to calendar month (self._min_month - 1 + i) % MONTHS_PER_YEAR
        actual_month_indices = [(self._min_month - 1 + sm) % MONTHS_PER_YEAR for sm in seasonal_months]
        month_labels = [month_names[idx] for idx in actual_month_indices]

        # Assign month names as coordinates
        data_array = data_array.assign_coords(month_name=('seasonal_month', month_labels))
        # Rename season_idx to year for clarity
        data_array = data_array.rename({'season_idx': 'year', 'seasonal_month': 'month'})

        if drop_first_year:
            data_array = data_array.isel(year=slice(1, None))
        return data_array

    def _create_and_populate_base_array(self, feature_name: str) -> tuple[np.ndarray, dict[str, int]]:
        """
        Create base array (locations, seasons, MONTHS_PER_YEAR) and populate from DataFrame.

        Returns:
            Tuple of (data_array, location_to_idx mapping)
        """
        locations = self._df['location'].unique()
        n_locations = len(locations)
        n_seasons = self._df['season_idx'].nunique()
        data_array = np.full((n_locations, n_seasons, MONTHS_PER_YEAR), np.nan)
        location_to_idx = {loc: idx for idx, loc in enumerate(locations)}

        # Populate base array from DataFrame
        for _, row in self._df.iterrows():
            loc_idx = location_to_idx[row['location']]
            season_idx = row['season_idx']
            month_idx = row['seasonal_month']
            data_array[loc_idx, season_idx, month_idx] = row[feature_name]

        return data_array, location_to_idx

    def _apply_right_padding(self, data_array: np.ndarray, feature_name: str, location_to_idx: dict[str, int]) -> np.ndarray:
        """
        Apply right padding by copying early months from next season to end of current season.

        This extends the seasonal array beyond MONTHS_PER_YEAR months to accommodate predictions
        that extend past the end of the disease year.
        """
        if not self._pad_right:
            return data_array

        n_locations, n_seasons = data_array.shape[0], data_array.shape[1]
        pad_array = np.full((n_locations, n_seasons, self._pad_right), np.nan)

        # Copy early months from next season to pad current season
        for _, row in self._df.iterrows():
            loc_idx = location_to_idx[row['location']]
            season_idx = row['season_idx']
            month_idx = row['seasonal_month']
            if month_idx < self._pad_right and season_idx > 0:
                pad_array[loc_idx, season_idx-1, month_idx] = row[feature_name]

        logger.info(f"Padding {self._pad_right} months to the right")
        logger.info(f"Before right pad: data_array.shape = {data_array.shape}, pad_array.shape = {pad_array.shape}")
        result = np.concatenate([data_array, pad_array], axis=-1)
        logger.info(f"After right pad: data_array.shape = {result.shape}")
        return result

    def _apply_left_padding(self, data_array: np.ndarray, feature_name: str, location_to_idx: dict[str, int]) -> np.ndarray:
        """
        Apply left padding by copying late months from previous season to start of current season.

        This ensures we have enough historical context for lagged features.
        """
        if not self._pad_left:
            return data_array

        n_locations, n_seasons = data_array.shape[0], data_array.shape[1]
        left_pad_array = np.full((n_locations, n_seasons, self._pad_left), np.nan)

        # Copy late months from previous season to start of next season
        for _, row in self._df.iterrows():
            loc_idx = location_to_idx[row['location']]
            season_idx = row['season_idx']
            month_idx = row['seasonal_month']
            if month_idx + self._pad_left >= MONTHS_PER_YEAR and season_idx < n_seasons - 1:
                left_pad_array[loc_idx, season_idx+1, month_idx+self._pad_left-MONTHS_PER_YEAR] = row[feature_name]

        logger.info(f"Padding {self._pad_left} months to the left")
        logger.info(f"Before left pad: data_array.shape = {data_array.shape}, left_pad_array.shape = {left_pad_array.shape}")
        result = np.concatenate([left_pad_array, data_array], axis=-1)
        logger.info(f"After left pad: data_array.shape = {result.shape}")
        return result

    def _drop_first_incomplete_year(self, data_array: np.ndarray) -> np.ndarray:
        """Drop first year which may be incomplete."""
        logger.info(f"Before dropping first year: data_array.shape = {data_array.shape}")
        result = data_array[:, 1:]
        logger.info(f"After dropping first year: result.shape = {result.shape}")
        return result

    def __getitem__(self, feature_name) -> np.ndarray:
        """
        Extract feature as array with shape (locations, seasons, months).

        Orchestrates the data transformation pipeline:
        1. Create and populate base array
        2. Apply padding (left and right)
        3. Drop first incomplete year
        """
        data_array, location_to_idx = self._create_and_populate_base_array(feature_name)
        data_array = self._apply_right_padding(data_array, feature_name, location_to_idx)
        data_array = self._apply_left_padding(data_array, feature_name, location_to_idx)
        result = self._drop_first_incomplete_year(data_array)
        return result


def test_seasonal_transform(df: pd.DataFrame):
    df['y'] = np.log1p(df['disease_cases'])
    st = SeasonalTransform(df)
    pivoted = st['y']
    assert pivoted.shape == (7, 13, MONTHS_PER_YEAR), pivoted

#def test_nepal_min(nepal_data: pd.DataFrame):
#    SeasonalTransform(nepal_data)

def test_xarray(colombia_df: pd.DataFrame):
    import altair
    df = colombia_df
    df['y'] = np.log1p(df['disease_cases'])
    y = SeasonalTransform(df).get_xarray('y')
    mean_y = y.mean(dim='seasonal_month')
    temp = SeasonalTransform(df).get_xarray('mean_temperature')
    corr = xarray.corr(mean_y, temp, dim='season_idx')
    corr_df = corr.to_dataframe(name='correlation').reset_index()

    # Add month names from the coordinate
    month_name_map = dict(zip(y.coords['seasonal_month'].values, y.coords['month_name'].values))
    corr_df['month_name'] = corr_df['seasonal_month'].map(month_name_map)

    chart = altair.Chart(corr_df).mark_bar().encode(
        x=altair.X('month_name:O', title='Month', sort=list(y.coords['month_name'].values)),
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