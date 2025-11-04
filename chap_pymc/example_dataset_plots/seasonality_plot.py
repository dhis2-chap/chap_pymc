import altair as alt
import numpy as np
import pandas as pd

from chap_pymc.dataset_plots import DatasetPlot


class SeasonalityPlot(DatasetPlot):

    def plot(self):
        data = self.data()
        chart = alt.Chart(data).mark_line().encode(
            x='month:O',
            y='y:Q',
            color='year:N',
            facet='location:N',
            tooltip=['location:N', 'feature:N', 'month:O', 'value:Q']
        ).properties(
            title='Seasonality Plot'
        )
        return chart

    def data(self):
        data = self._df
        data['date'] = pd.to_datetime(data['time_period'] + '-01')
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        data['y'] = np.log1p(data['disease_cases'])
        return data


class MeanSeasonalityPlot(SeasonalityPlot):

    def plot(self):
        data = self.data()
        summary = data.groupby(['location', 'month']).agg(
            mean_y=('y', 'mean'),
            std_y=('y', 'std')
        ).reset_index()
        min_index = summary.groupby('location').agg(min_idx=('mean_y', 'idxmin')).reset_index()
        print(summary[['month', 'location']].iloc[min_index['min_idx']])  # Set the month with minimum mean_y to 0 for each location
        summary['y_lower'] = summary['mean_y'] - summary['std_y']
        summary['y_upper'] = summary['mean_y'] + summary['std_y']

        error_bars = alt.Chart(summary).mark_errorbar(extent='stdev').encode(
            x=alt.X('month:O', title='Month'),
            y=alt.Y('mean_y:Q', title='Mean log1p(disease_cases)'),
            yError=alt.YError('std_y:Q'),
            tooltip=['location:N', 'month:O', 'mean_y:Q', 'std_y:Q']
        )

        points = alt.Chart(summary).mark_circle(size=60, color='black').encode(
            x=alt.X('month:O', title='Month'),
            y=alt.Y('mean_y:Q', title='Mean log1p(disease_cases)'),
            tooltip=['location:N', 'month:O', 'mean_y:Q', 'std_y:Q']
        )

        chart = (error_bars + points).facet(
            column=alt.Facet('location:N', title='Location')
        ).properties(
            title='Mean Seasonality of log1p(disease_cases) with Std Error Bars'
        )

        return chart

def test_seasonality_plot(viet_begin_season):
    chart = SeasonalityPlot(viet_begin_season).plot()
    chart.show()

def test_mean_seasonality_plot(viet_begin_season):
    chart = MeanSeasonalityPlot(viet_begin_season).plot()
    chart.show()
