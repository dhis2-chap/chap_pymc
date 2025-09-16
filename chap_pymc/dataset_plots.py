from abc import ABC, abstractmethod
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import pytest
from altair import FacetChart
alt.data_transformers.enable('vegafusion')
alt.renderers.enable('browser')
#alt.renderers.enable('notebook')

class DatasetPlot(ABC):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    @abstractmethod
    def plot(self) -> alt.Chart:
        ...

    @abstractmethod
    def data(self):
        ...

class StandardizedFeaturePlot(DatasetPlot):
    '''
    This plot shows standardized(zero mean, unit variance) features over time for different locations.
    It includes a log1p transformation of the disease incidence rate (disease_cases/population)
    This shows how different features correlate over time and location.
    '''
    
    def __init__(self, df: pd.DataFrame, selected_features=None):
        super().__init__(df)
        self.selected_features = selected_features
    def _standardize(self, col: np.array) -> np.array:
        # Handle NaN values properly
        mean_val = np.nanmean(col)
        std_val = np.nanstd(col)
        if std_val == 0:
            return col - mean_val  # Return zero-centered values when std is 0
        return (col - mean_val) / std_val

    def data(self) -> pd.DataFrame:
        df = self._df.copy()
        colnames = list(self._get_colnames())
        base_df = df[['time_period', 'location']].copy()
        
        # Add log1p of disease incidence rate if population column exists
        if 'population' in df.columns:
            df['log1p'] = np.log1p(df['disease_cases'] / df['population'])
            colnames.append('log1p')
        else:
            # Fallback to just log1p of disease cases
            df['log1p'] = np.log1p(df['disease_cases'])
            colnames.append('log1p')
        
        dfs = []
        for colname in colnames:
            if colname in df.columns:
                new_df = base_df.copy()
                new_df['value'] = self._standardize(df[colname].values)
                new_df['feature'] = colname
                dfs.append(new_df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            # Return empty dataframe with correct structure
            return pd.DataFrame(columns=['time_period', 'location', 'value', 'feature'])

    def _get_colnames(self) -> filter:
        colnames = filter(lambda name: name not in ('disease_cases', 'location', 'time_period') and not name.startswith('Unnamed'), self._df.columns)
        colnames = filter(lambda name: self._df[name].dtype.name in ('float64', 'int64'), colnames)
        return colnames

    def plot(self) -> FacetChart:
        data = self.data()
        
        # Filter data based on selected features if specified
        if self.selected_features is not None:
            data = data[data['feature'].isin(self.selected_features)]
        
        # Convert time_period to proper datetime format
        data['date'] = pd.to_datetime(data['time_period'] + '-01')


        checkbox_selection = alt.selection_point(
            fields=['feature'],
            toggle='true'
        )

        # Create legend that acts as checkboxes
        legend_chart = alt.Chart(data).mark_circle(size=100).add_params(
            checkbox_selection
        ).encode(
            y=alt.Y('feature:N', axis=alt.Axis(orient='right', title='Select Features')),
            color=alt.condition(
                checkbox_selection,
                alt.Color('feature:N', legend=None),
                alt.value('lightgray')
            ),
            tooltip=['feature:N']
        ).properties(
            width=100,
            title="Click to select/deselect"
        )

        # Main chart with filtering
        main_chart = alt.Chart(data).add_params(
            alt.selection_interval(bind='scales', encodings=['x'])
        ).mark_line(
            point=False, strokeWidth=2
        ).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('value:Q', title='Standardized Value'),
            color=alt.Color('feature:N', legend=alt.Legend(title="Feature")),
            opacity=alt.condition(checkbox_selection, alt.value(1.0), alt.value(0.1)),
            tooltip=['date:T', 'feature:N', 'value:Q', 'location:N']
        ).facet(
            facet=alt.Facet('location:N', title='Location'),
            columns=3
        ).resolve_scale(
            y='shared'
        )
        return alt.hconcat(legend_chart, main_chart).resolve_legend(
            color="independent"
        ).properties(
            title="Multiple Feature Selection (Click legend items to toggle)"
        )


@pytest.fixture
def df() -> pd.DataFrame:
    p = Path(__file__).parent.parent/ 'test_data' / 'training_data.csv'
    return pd.read_csv(p)



def test_standardized_feature_plot(df: pd.DataFrame):
    plotter = StandardizedFeaturePlot(df)
    data = plotter.data()
    print(data.head())
    print(f"Data shape: {data.shape}")
    print(f"Features: {data['feature'].unique()}")
    assert 'value' in data.columns
    assert 'feature' in data.columns
    assert 'location' in data.columns
    assert 'time_period' in data.columns
    
    chart = plotter.plot()
    chart.save('standardized_feature_plot.html')
    print("Chart saved to standardized_feature_plot.html")
