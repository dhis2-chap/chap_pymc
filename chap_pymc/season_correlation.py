import altair as alt
import numpy as np
import pandas as pd
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class SeasonCorrelationPlot:
    def __init__(self, dataset: DataSet):
        self.dataset = dataset

    def plot(self):
        correlation_data = []

        for location, ds in self.dataset.items():
            df = ds.to_pandas()
            df['y'] = np.log1p(df['disease_cases'])
            df['lag_1'] = df['y'].shift(1)
            df['lag_2'] = df['y'].shift(2)
            df['lag_3'] = df['y'].shift(3)
            df['month'] = df['time_period'].dt.month

            # Calculate correlation for each month and each lag
            for lag in [1, 2, 3]:
                lag_col = f'lag_{lag}'
                for month in range(1, 13):
                    month_data = df[df['month'] == month]
                    if len(month_data) > 1:  # Need at least 2 points for correlation
                        # Remove NaN values
                        valid_data = month_data[['y', lag_col]].dropna()
                        if len(valid_data) > 1:
                            corr = valid_data['y'].corr(valid_data[lag_col])
                            correlation_data.append({
                                'location': location,
                                'month': month,
                                'lag': lag,
                                'correlation': corr
                            })

        corr_df = pd.DataFrame(correlation_data)

        # Calculate mean and std for error bars
        summary_df = corr_df.groupby(['month', 'lag'])['correlation'].agg(['mean', 'std']).reset_index()
        summary_df['correlation_lower'] = summary_df['mean'] - summary_df['std']
        summary_df['correlation_upper'] = summary_df['mean'] + summary_df['std']

        # Create error bar plot with faceting by lag
        error_bars = alt.Chart(summary_df).mark_errorbar(extent='stdev').encode(
            x=alt.X('month:O', title='Month'),
            y=alt.Y('mean:Q', title='Mean Correlation'),
            yError=alt.YError('std:Q'),
            tooltip=['month', 'lag', 'mean', 'std']
        )

        points = alt.Chart(summary_df).mark_circle(size=60, color='black').encode(
            x=alt.X('month:O', title='Month'),
            y=alt.Y('mean:Q', title='Mean Correlation'),
            tooltip=['month', 'lag', 'mean', 'std']
        )

        chart = (error_bars + points).facet(
            column=alt.Facet('lag:O', title='Lag')
        ).properties(
            title='Mean Correlation between y and lags by Month (with std error bars)'
        )

        return chart



def test_season_correlation_plot(thailand_ds: DataSet):
    plot = SeasonCorrelationPlot(thailand_ds)
    chart = plot.plot()
    chart.save('season_correlation_plot.html')
    #assert isinstance(chart, altair.Chart) or isinstance(chart, altair.FacetChart) or isinstance(chart, altair.HConcatChart)


