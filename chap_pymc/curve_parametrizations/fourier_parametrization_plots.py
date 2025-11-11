"""Plotting functions for Fourier parametrization visualizations"""
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chap_pymc.correlation_plots import CorrelationBarPlot


def plot_faceted_predictions(y, mu_mean, mu_lower, mu_upper, coords, output_file='fourier_parametrization_fit.png'):
    """
    Create a faceted plot with rows=years, cols=locations showing observed vs predicted values.

    Args:
        y: xarray.DataArray with observed data, dims=(location, epi_year, epi_offset)
        mu_mean: Posterior mean predictions, shape (location, epi_year, epi_offset)
        mu_lower: Lower credible interval, shape (location, epi_year, epi_offset)
        mu_upper: Upper credible interval, shape (location, epi_year, epi_offset)
        coords: Dictionary with 'location', 'epi_year', 'epi_offset' coordinates
        output_file: Path to save the plot
    """
    n_locations = len(coords['location'])
    n_years = len(coords['epi_year'])

    fig, axes = plt.subplots(n_years, n_locations, figsize=(4 * n_locations, 3 * n_years))

    # Handle single location or single year case
    if n_years == 1 and n_locations == 1:
        axes = np.array([[axes]])
    elif n_years == 1:
        axes = axes.reshape(1, -1)
    elif n_locations == 1:
        axes = axes.reshape(-1, 1)

    months = np.arange(12)

    for year_idx in range(n_years):
        for loc_idx in range(n_locations):
            ax = axes[year_idx, loc_idx]

            # Observed data
            y_obs = y.values[loc_idx, year_idx, :]
            ax.plot(months, y_obs, 'o-', alpha=0.7, label='Observed', color='C0')

            # Posterior mean
            y_pred = mu_mean.values[loc_idx, year_idx, :]
            ax.plot(months, y_pred, '--', alpha=0.7, label='Predicted', color='C1')

            # Credible interval
            lower = mu_lower.values[loc_idx, year_idx, :]
            upper = mu_upper.values[loc_idx, year_idx, :]
            ax.fill_between(months, lower, upper, alpha=0.2, color='C1')

            # Set title for top row (location names)
            if year_idx == 0:
                ax.set_title(f'Location {loc_idx}')

            # Set ylabel for first column (year labels)
            if loc_idx == 0:
                ax.set_ylabel(f'Year {year_idx}\nValue')

            # Set xlabel for bottom row
            if year_idx == n_years - 1:
                ax.set_xlabel('Month')

            ax.grid(True, alpha=0.3)

            # Only show legend on first subplot
            if year_idx == 0 and loc_idx == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")


def plot_vietnam_faceted_predictions(
    y, mu_mean, mu_lower, mu_upper, coords,
    n_locations_to_plot=20,
    output_file='vietnam_fourier_fit.png'
):
    """
    Create a faceted plot for Vietnam data with rows=years, cols=locations.

    Args:
        y: xarray.DataArray with observed data, dims=(location, epi_year, epi_offset)
        mu_mean: xarray.DataArray with posterior mean predictions, dims=(location, epi_year, epi_offset)
        mu_lower: xarray.DataArray with lower credible interval, dims=(location, epi_year, epi_offset)
        mu_upper: xarray.DataArray with upper credible interval, dims=(location, epi_year, epi_offset)
        coords: Dictionary with 'location', 'epi_year', 'epi_offset' coordinates
        n_locations_to_plot: Maximum number of locations to plot
        output_file: Path to save the plot
    """
    # Get coordinate values from DataArrays
    locations = y.location.values[:n_locations_to_plot]
    years = y.epi_year.values
    n_locations = len(locations)
    n_years = len(years)

    fig, axes = plt.subplots(n_years, n_locations,
                            figsize=(4 * n_locations, 2.5 * n_years),
                            sharey=True)

    # Handle single location or single year case
    if n_years == 1 and n_locations == 1:
        axes = np.array([[axes]])
    elif n_years == 1:
        axes = axes.reshape(1, -1)
    elif n_locations == 1:
        axes = axes.reshape(-1, 1)

    for year_idx, year in enumerate(years):
        for loc_idx, location in enumerate(locations):
            ax = axes[year_idx, loc_idx]

            # Select data using .sel() with coordinate values
            y_obs = y.sel(location=location, epi_year=year)
            y_pred = mu_mean.sel(location=location, epi_year=year)
            lower = mu_lower.sel(location=location, epi_year=year)
            upper = mu_upper.sel(location=location, epi_year=year)

            # Use epi_offset coordinate for x-axis
            months = y_obs.epi_offset.values

            # Plot observed data
            ax.plot(months, y_obs.values, 'o-', alpha=0.7, label='Observed', color='C0', markersize=3)

            # Plot posterior mean
            ax.plot(months, y_pred.values, '--', alpha=0.7, label='Predicted', color='C1')

            # Plot credible interval
            ax.fill_between(months, lower.values, upper.values, alpha=0.2, color='C1')

            # Set title for top row (location names)
            if year_idx == 0:
                ax.set_title(f'{location}', fontsize=9)

            # Set ylabel for first column (year labels)
            if loc_idx == 0:
                ax.set_ylabel(f'Year {year}\nLog(Cases)', fontsize=8)

            # Set xlabel for bottom row
            if year_idx == n_years - 1:
                ax.set_xlabel('Month', fontsize=8)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            ax.set_ylim(-2.5, 2.5)

            # Only show legend on first subplot
            if year_idx == 0 and loc_idx == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    print(f"Plotted {n_locations} locations across {n_years} years")


class FourierParameterCorrelationPlot(CorrelationBarPlot):
    """
    Bar plot showing correlations between Fourier parameters and temperature features.

    For each location, computes correlations across years between:
    - Parameters: baseline, A1, A2, A3, ... (harmonic amplitudes)
    - Features: temperature lags from model input
    """

    def __init__(self, idata, model_input, n_harmonics: int):
        """
        Initialize with inference data and model input.

        Args:
            idata: ArviZ InferenceData with posterior samples
            model_input: ModelInput with X features (xarray DataArray)
            n_harmonics: Number of harmonics used in the Fourier model
        """
        self.idata = idata
        self.model_input = model_input
        self.n_harmonics = n_harmonics

    def data(self) -> pd.DataFrame:
        """
        Compute correlations between Fourier parameters and features per location.

        Returns:
            DataFrame with columns: ['location', 'correlation', 'feature', 'outcome', 'combination']
        """
        # Extract posterior means for parameters
        posterior = self.idata.posterior

        # Get features from model_input.X
        # Shape: (location, year, feature)
        X = self.model_input.X
        locations = X.coords['location'].values
        features = X.coords['feature'].values

        correlations = []

        # Handle A parameter (location, year, harmonic) where harmonic includes h=0 (baseline)
        A_posterior = posterior['A']
        A_mean = A_posterior.mean(dim=['chain', 'draw'])  # (location, year, harmonic)

        for loc_idx, location in enumerate(locations):
            # For each harmonic (including h=0 which is the baseline)
            for h in range(self.n_harmonics + 1):  # +1 to include h=0
                A_values = A_mean.values[loc_idx, :, h]  # Get this location's years for harmonic h

                # Label h=0 as 'baseline', h>0 as 'A1', 'A2', etc.
                param_name = 'baseline' if h == 0 else f'A{h}'

                for feat_idx, feature in enumerate(features):
                    # Get feature values for this location across years
                    feat_values = X.values[loc_idx, :, feat_idx]

                    # Compute correlation across years
                    valid_mask = ~(np.isnan(A_values) | np.isnan(feat_values))
                    if valid_mask.sum() > 1:  # Need at least 2 points
                        corr = np.corrcoef(A_values[valid_mask], feat_values[valid_mask])[0, 1]
                    else:
                        corr = np.nan

                    correlations.append({
                        'location': str(location),
                        'outcome': param_name,
                        'feature': str(feature),
                        'correlation': corr,
                        'combination': f'{feature}_vs_{param_name}'
                    })

        return pd.DataFrame(correlations)

    def plot(self) -> alt.FacetChart:
        """Create bar plot using base class with custom parameters."""
        return super().plot(
            title="Fourier Parameter - Temperature Feature Correlations",
            subtitle="Correlation between estimated Fourier parameters and temperature features across years, by location. Red bars indicate negative correlation, blue bars indicate positive correlation.",
            x_col='outcome',
            x_title='Fourier Parameter',
            row_facet='feature',
            row_title='Temperature Feature'
        )


def plot_parameter_feature_correlations(idata, model_input, n_harmonics: int,
                                       output_file: str = 'parameter_feature_correlations.html'):
    """
    Create and save a bar plot of parameter-feature correlations.

    Args:
        idata: ArviZ InferenceData with posterior samples
        model_input: ModelInput with X features (xarray DataArray)
        n_harmonics: Number of harmonics used in the Fourier model
        output_file: Path to save the plot (HTML format for Altair)

    Returns:
        Altair FacetChart
    """
    plotter = FourierParameterCorrelationPlot(idata, model_input, n_harmonics)
    chart = plotter.plot()
    chart.save(output_file)
    print(f"Correlation plot saved to {output_file}")
    return chart
