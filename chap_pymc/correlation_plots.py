"""Generic correlation plotting classes using Altair"""
from abc import ABC, abstractmethod

import altair as alt
import pandas as pd


class CorrelationBarPlot(ABC):
    """
    Abstract base class for creating correlation bar plots.

    Subclasses should implement the data() method to return a DataFrame with:
    - location: Location identifier
    - correlation: Correlation coefficient value
    - feature: Feature name
    - outcome: Outcome/parameter name
    - combination: String combining feature and outcome (for faceting)
    """

    @abstractmethod
    def data(self) -> pd.DataFrame:
        """
        Compute correlations and return DataFrame with required columns.

        Returns:
            DataFrame with columns: ['location', 'correlation', 'feature', 'outcome', 'combination']
        """
        pass

    def plot(self, title: str = "Correlation Analysis",
             subtitle: str = "Correlation coefficients by location and feature combination",
             x_col: str = 'outcome',
             x_title: str = 'Parameter',
             row_facet: str = 'feature',
             row_title: str = 'Feature') -> alt.FacetChart:
        """
        Create a bar plot of correlations faceted by location and feature.

        Args:
            title: Main title for the plot
            subtitle: Subtitle for the plot
            x_col: Column to use for x-axis (default: 'outcome')
            x_title: Title for x-axis
            row_facet: Column to use for row faceting (default: 'feature')
            row_title: Title for row facet

        Returns:
            Altair FacetChart with bar plots
        """
        df = self.data()

        return alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{x_col}:N', title=x_title),
            y=alt.Y('correlation:Q', title='Correlation Coefficient'),
            color=alt.Color('correlation:Q',
                          scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                          title='Correlation'),
            tooltip=['location:N', f'{x_col}:N', f'{row_facet}:N', 'correlation:Q']
        ).facet(
            column=alt.Column('location:N', title='Location'),
            row=alt.Row(f'{row_facet}:N', title=row_title)
        ).properties(
            title={
                "text": title,
                "subtitle": subtitle
            }
        )
