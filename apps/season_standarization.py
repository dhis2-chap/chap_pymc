import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).parent.parent))
from per_season.yearly_pattern import create_data_arrays
from chap_pymc.dataset_plots import StandardizedFeaturePlot

alt.data_transformers.enable('json')

st.set_page_config(page_title="Season Standardization Analysis", layout="wide")

def main():
    st.title("Season Standardization Analysis")
    st.markdown("Interactive visualization of seasonal disease patterns and normalization")
    
    # Try to auto-load sample data
    sample_path = "/Users/knutdr/Downloads/dataset2/training_data.csv"
    df = None
    
    if Path(sample_path).exists():
        try:
            df = pd.read_csv(sample_path)
            st.success(f"Sample data auto-loaded! Shape: {df.shape}")
            st.info("Analysis generated automatically with sample data. You can upload your own data below to override.")
        except Exception as e:
            st.warning(f"Could not load sample data: {str(e)}")
    
    # File uploader for custom data
    uploaded_file = st.file_uploader("Upload your own CSV file to override sample data", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Custom data loaded successfully! Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error loading uploaded data: {str(e)}")
            df = None
    
    # Generate analysis if we have data
    if df is not None:
        with st.spinner("Processing data and generating plots..."):
            try:
                analyze_data(df)
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)
    else:
        st.warning("No data available for analysis. Please check that sample data exists or upload a CSV file.")

def analyze_data(df: pd.DataFrame):
    st.subheader("Data Overview")
    st.dataframe(df.head(n=100))
    
    # Standardized Feature Plot from dataset_plots.py
    st.subheader("Standardized Feature Analysis")
    st.write("All features standardized to have mean=0 and std=1 for comparison")
    
    try:
        plotter = StandardizedFeaturePlot(df)
        full_data = plotter.data()
        
        if not full_data.empty:
            # Get available features for selection
            available_std_features = full_data['feature'].unique()
            
            st.write("Select features to display:")
            selected_std_features = []
            
            # Create checkboxes for feature selection
            std_cols = st.columns(min(4, len(available_std_features)))  # Max 4 columns
            for i, feature in enumerate(available_std_features):
                with std_cols[i % len(std_cols)]:
                    if st.checkbox(feature.replace('_', ' ').title(), value=True, key=f"std_feature_{feature}"):
                        selected_std_features.append(feature)
            
            if selected_std_features:
                # Filter data to only include selected features
                filtered_data = full_data[full_data['feature'].isin(selected_std_features)]
                
                # Convert time_period to proper datetime format
                filtered_data['date'] = pd.to_datetime(filtered_data['time_period'] + '-01')
                
                # Create the chart with filtered data
                standardized_chart = alt.Chart(filtered_data).add_params(
                    alt.selection_interval(bind='scales', encodings=['x'])
                ).mark_line(
                    point=False, strokeWidth=2
                ).encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('value:Q', title='Standardized Value'),
                    color=alt.Color('feature:N', legend=alt.Legend(title="Feature")),
                    tooltip=['date:T', 'feature:N', 'value:Q', 'location:N']
                ).facet(
                    facet=alt.Facet('location:N', title='Location'),
                    columns=3
                ).resolve_scale(
                    y='shared'
                )
                
                st.altair_chart(standardized_chart, use_container_width=True)
            else:
                st.warning("Please select at least one feature to display.")
        else:
            st.warning("No standardized feature data available.")
            
    except Exception as e:
        st.error(f"Error creating standardized feature plot: {str(e)}")
        st.exception(e)
    
    # Feature Analysis - Time series of all features
    st.subheader("Rescaled Feature Analysis Over Time")
    st.write("Features rescaled to [0-1] range for comparison")
    
    # Check which features are available
    available_features = []
    feature_columns = {
        'disease_cases': 'Disease Cases',
        'rainfall': 'Rainfall', 
        'temperature': 'Temperature',
        'population': 'Population'
    }
    
    for col, label in feature_columns.items():
        if col in df.columns:
            available_features.append((col, label))
    
    if available_features:
        # Feature selection with checkboxes
        st.write("Select features to display:")
        selected_features = []
        cols = st.columns(len(available_features))
        
        for i, (col, label) in enumerate(available_features):
            with cols[i]:
                if st.checkbox(label, value=True, key=f"feature_{col}"):
                    selected_features.append((col, label))
        
        if selected_features:
            # Prepare data for plotting
            plot_data = df.copy()
            
            # Add date column for time series
            plot_data['date'] = pd.to_datetime(plot_data['time_period'] + '-01')
            
            # Rescale selected features to [0, 1] range
            scaler = MinMaxScaler()
            rescaled_data = []
            
            for location in plot_data['location'].unique():
                location_data = plot_data[plot_data['location'] == location].copy()
                location_data = location_data.sort_values('date')
                
                # Get feature data
                feature_matrix = location_data[[col for col, _ in selected_features]].values
                
                # Only rescale if we have valid data
                if not np.isnan(feature_matrix).all():
                    # Handle NaN values by using nanmin/nanmax for scaling
                    valid_data = feature_matrix[~np.isnan(feature_matrix).any(axis=1)]
                    if len(valid_data) > 0:
                        scaler.fit(valid_data)
                        # Apply scaling, preserving NaN values
                        for i in range(len(feature_matrix)):
                            if not np.isnan(feature_matrix[i]).any():
                                continue
                                feature_matrix[i] = scaler.transform([feature_matrix[i]])[0]
                
                # Create long format data
                for i, (col, label) in enumerate(selected_features):
                    for idx, (_, row) in enumerate(location_data.iterrows()):
                        rescaled_data.append({
                            'location': location,
                            'date': row['date'],
                            'time_period': row['time_period'],
                            'feature': label,
                            'value': feature_matrix[idx, i] if idx < len(feature_matrix) else np.nan,
                            'original_value': row[col]
                        })
            
            plot_df = pd.DataFrame(rescaled_data)
            plot_df = plot_df.dropna(subset=['value'])
            
            if not plot_df.empty:
                # Create time series plot faceted by location
                time_series_chart = alt.Chart(plot_df).mark_line(
                    point=True, strokeWidth=2
                ).encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('value:Q', title='Rescaled Value (0-1)', scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('feature:N', legend=alt.Legend(title="Feature")),
                    tooltip=['location:N', 'date:T', 'feature:N', 'value:Q', 'original_value:Q']
                ).properties(
                    width=400,
                    height=200
                ).facet(
                    column=alt.Column('location:N', title='Location'),
                    columns=2
                ).resolve_scale(
                    x='shared'
                )
                
                st.altair_chart(time_series_chart, use_container_width=True)
                
                st.info("All features are rescaled to [0-1] range for comparison. Hover over points to see original values.")
            else:
                st.warning("No valid data to display for selected features.")
        else:
            st.warning("Please select at least one feature to display.")
    else:
        st.warning("No recognizable feature columns found in the data.")
    
    # Prepare the DataFrame with additional columns
    df = df.copy()
    df['log1p'] = np.log1p(df['disease_cases']/df['population'])
    df['month'] = df['time_period'].apply(lambda x: int(x.split('-')[1])-1)
    df['year'] = df['time_period'].apply(lambda x: int(x.split('-')[0]))
    
    # Find minimum month
    means = ((month, group['disease_cases'].mean()) for month, group in df.groupby('month'))
    min_month, val = min(means, key=lambda x: x[1])
    
    st.subheader(f"Analysis Details")
    st.write(f"Minimum disease cases month: {min_month} (avg cases: {val:.2f})")
    
    horizon = st.slider("Forecast Horizon", min_value=1, max_value=12, value=3)
    
    # Add season column (month - min_month, wrapped)
    assert df['month'].max()==11
    df['season'] = ((df['month'] - min_month) % 12)
    
    # Create season_idx (year index starting from min_month)
    df['season_idx'] = df['year'] + df['month']//12
    df['season_idx'] -= df['season_idx'].min()
    # Calculate means and stds for each location x season_idx
    season_stats = df.groupby(['location', 'season_idx'])['log1p'].agg(['mean', 'std']).reset_index()
    season_stats.columns = ['location', 'season_idx', 'season_mean', 'season_std']
    
    # Merge back to main dataframe
    df = df.merge(season_stats, on=['location', 'season_idx'], how='left')
    
    # Add normalized values column: (log1p - season_mean) / season_std
    df['log1p_location_normalized'] = df['log1p'] - df['season_mean']


    df['log1p_normalized'] = (df['log1p_location_normalized']) / df['season_std'].clip(lower=0.001)
    
    # Calculate seasonal pattern means and stds for each location
    seasonal_pattern_stats = df.groupby(['location', 'season'])['log1p_normalized'].agg(['mean', 'std']).reset_index()
    seasonal_pattern_stats.columns = ['location', 'season', 'pattern_mean', 'pattern_std']
    
    # Merge seasonal pattern stats
    df = df.merge(seasonal_pattern_stats, on=['location', 'season'], how='left')
    
    # Add fully normalized values: (log1p_normalized - pattern_mean) / pattern_std
    df['fully_normalized'] = ((df['log1p_normalized'] - df['pattern_mean']) / 
                             df['pattern_std'].clip(lower=0.001))
    
    # Estimate linear trends for fully_normalized over season for each location
    df['linear_trend'] = np.nan
    df['fully_normalized_detrended'] = np.nan
    
    for location in df['location'].unique():
        for season_idx in df['season_idx'].unique():
            iter_mask = (df['location'] == location) & (df['season_idx'] == season_idx)
            location_data = df[iter_mask].copy()

            # Prepare data for linear regression on fully_normalized data
            X = location_data['season'].values.reshape(-1, 1)
            y = location_data['fully_normalized'].values

            # Remove any NaN values
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() > 1:  # Need at least 2 points for linear regression
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]

                # Fit linear regression
                model = LinearRegression()
                model.fit(X_valid, y_valid)

                # Predict linear trend for all seasons
                trend_predictions = model.predict(X)
                df.loc[iter_mask, 'linear_trend'] = trend_predictions
                df.loc[iter_mask, 'fully_normalized_detrended'] = location_data['fully_normalized'] - trend_predictions

    
    st.subheader("Enhanced Data Overview")
    st.dataframe(df[['location', 'time_period', 'season', 'season_idx', 'log1p', 
                     'season_mean', 'season_std', 'log1p_normalized', 
                     'pattern_mean', 'pattern_std', 'fully_normalized', 'linear_trend', 'fully_normalized_detrended']].head())
    
    locations = df['location'].unique()
    
    # Create long format DataFrame for normalization stages
    st.subheader("Comparison: Unnormalized → Location Normalized → Standardized → Pattern Normalized → Detrended")
    
    # Reshape to long format
    df_long = pd.melt(
        df, 
        id_vars=['location', 'time_period', 'season', 'season_idx'],
        value_vars=['log1p', 'log1p_location_normalized', 'log1p_normalized', 'fully_normalized', 'fully_normalized_detrended'],
        var_name='normalization_stage',
        value_name='value'
    )
    
    # Create more readable stage names
    stage_labels = {
        'log1p': 'Unnormalized',
        'log1p_location_normalized': 'Location Normalized', 
        'log1p_normalized': 'Standardized',
        'fully_normalized': 'Pattern Normalized',
        'fully_normalized_detrended': 'Detrended'
    }
    df_long['stage_label'] = df_long['normalization_stage'].map(stage_labels)
    
    # Create single chart with column faceting
    faceted_chart = alt.Chart(df_long).mark_line(
        point=True, opacity=0.7
    ).encode(
        x=alt.X('season:O', title='Season', scale=alt.Scale(domain=list(range(12)))),
        y=alt.Y('value:Q', title='Value'),
        color=alt.Color('season_idx:N', legend=alt.Legend(title="Season Year")),
        tooltip=['season:O', 'value:Q', 'season_idx:N', 'time_period:N', 'stage_label:N']
    ).properties(
        width=200,
        height=150
    ).facet(
        column=alt.Column('stage_label:N', title='Normalization Stage', sort=['Unnormalized', 'Location Normalized', 'Standardized', 'Pattern Normalized', 'Detrended']),
        row=alt.Row('location:N', title='Location')
    ).resolve_scale(
        y='independent'
    )
    
    st.altair_chart(faceted_chart, use_container_width=True)
    
    # Standard deviation plot for each month/season by location
    st.subheader("Monthly Standard Deviations by Location")
    
    # Calculate standard deviations for each location and season
    monthly_std = df.groupby(['location', 'season']).agg({
        'log1p': 'std',
        'log1p_normalized': 'std', 
        'fully_normalized': 'std',
        'fully_normalized_detrended': 'std'
    }).reset_index()
    
    # Rename columns for clarity
    monthly_std.columns = ['location', 'season', 'log1p_std', 'normalized_std', 'pattern_std', 'detrended_std']
    
    # Create long format for the standard deviations
    std_long = pd.melt(
        monthly_std,
        id_vars=['location', 'season'],
        value_vars=['log1p_std', 'normalized_std', 'pattern_std', 'detrended_std'],
        var_name='std_type',
        value_name='std_value'
    )
    
    # Create readable labels
    std_labels = {
        'log1p_std': 'Log1p Std',
        'normalized_std': 'Normalized Std',
        'pattern_std': 'Pattern Norm Std',
        'detrended_std': 'Detrended Std'
    }
    std_long['std_label'] = std_long['std_type'].map(std_labels)
    
    # Create standard deviation chart
    std_chart = alt.Chart(std_long).mark_line(
        point=True, strokeWidth=2
    ).encode(
        x=alt.X('season:O', title='Season (Month from Minimum)', scale=alt.Scale(domain=list(range(12)))),
        y=alt.Y('std_value:Q', title='Standard Deviation'),
        color=alt.Color('std_label:N', legend=alt.Legend(title="Std Type")),
        tooltip=['season:O', 'std_value:Q', 'std_label:N', 'location:N']
    ).properties(
        width=200,
        height=150
    ).facet(
        row=alt.Row('location:N', title='Location')
    ).resolve_scale(
        y='independent'
    )
    
    st.altair_chart(std_chart, use_container_width=True)
    
    # Extract parameters for each location and year
    st.subheader("Parameter Analysis by Location and Year")
    
    # Create parameters DataFrame
    params_list = []
    
    for location in df['location'].unique():
        location_data = df[df['location'] == location].copy()
        
        for season_idx in location_data['season_idx'].unique():
            year_data = location_data[location_data['season_idx'] == season_idx].copy()
            
            if len(year_data) > 0:
                # Location parameter: mean of log1p for this location-year
                location_param = year_data['season_mean'].iloc[0] if not year_data['season_mean'].isna().all() else np.nan
                
                # Scale parameter: season_std for this location-year  
                scale_param = year_data['season_std'].iloc[0] if not year_data['season_std'].isna().all() else np.nan
                
                # Trend parameter: slope from linear regression (extract from linear_trend)
                if not year_data['linear_trend'].isna().all():
                    # Calculate slope from linear trend values
                    seasons = year_data['season'].values
                    trends = year_data['linear_trend'].values
                    valid_mask = ~np.isnan(trends)
                    
                    if valid_mask.sum() > 1:
                        # Fit to get slope
                        slope = (trends[valid_mask][-1] - trends[valid_mask][0]) / (seasons[valid_mask][-1] - seasons[valid_mask][0]) if seasons[valid_mask][-1] != seasons[valid_mask][0] else 0
                        trend_param = slope
                    else:
                        trend_param = np.nan
                else:
                    trend_param = np.nan
                
                params_list.append({
                    'location': location,
                    'season_idx': season_idx,
                    'location_param': location_param,
                    'scale_param': scale_param, 
                    'trend_param': trend_param
                })
    
    params_df = pd.DataFrame(params_list)
    
    # Remove rows with any NaN parameters and add log(scale)
    params_df_clean = params_df.dropna()
    
    # Add log(scale) parameter, handling zero/negative values
    params_df_clean['log_scale_param'] = np.log(params_df_clean['scale_param'].clip(lower=1e-10))
    
    st.write(f"Parameters extracted for {len(params_df_clean)} location-year combinations")
    st.dataframe(params_df_clean.head())
    
    # Create scatter plots for each parameter combination
    if len(params_df_clean) > 0:
        # First show histograms of each parameter
        st.subheader("Parameter Distributions")
        
        # Define parameters to plot
        histogram_params = [
            ('location_param', 'Location Parameter'),
            ('log_scale_param', 'Log(Scale) Parameter'),
            ('trend_param', 'Trend Parameter')
        ]
        
        # Create histograms in columns
        hist_cols = st.columns(len(histogram_params))
        
        for i, (param, title) in enumerate(histogram_params):
            with hist_cols[i]:
                hist_chart = alt.Chart(params_df_clean).mark_bar(
                    opacity=0.7,
                    color='steelblue'
                ).encode(
                    x=alt.X(f'{param}:Q', bin=alt.Bin(maxbins=20), title=title),
                    y=alt.Y('count():Q', title='Count'),
                    tooltip=['count():Q']
                ).properties(
                    title=f'{title} Distribution',
                    width=250,
                    height=200
                )
                
                st.altair_chart(hist_chart, use_container_width=True)
        
        st.subheader("2D Parameter Scatter Plots")
        
        # Define parameter combinations using log(scale)
        param_combinations = [
            ('location_param', 'log_scale_param', 'Location vs Log(Scale)'),
            ('location_param', 'trend_param', 'Location vs Trend'),
            ('log_scale_param', 'trend_param', 'Log(Scale) vs Trend')
        ]
        
        # Create scatter plots with faceting by location
        for i, (x_param, y_param, title) in enumerate(param_combinations):
            scatter_chart = alt.Chart(params_df_clean).mark_circle(
                size=100, opacity=0.7, color='steelblue'
            ).encode(
                x=alt.X(f'{x_param}:Q', title=x_param.replace('_param', '').replace('log_scale', 'Log(Scale)').title()),
                y=alt.Y(f'{y_param}:Q', title=y_param.replace('_param', '').replace('log_scale', 'Log(Scale)').title()),
                tooltip=['location:N', 'season_idx:N', f'{x_param}:Q', f'{y_param}:Q']
            ).properties(
                title=title,
                width=200,
                height=150
            ).facet(
                column=alt.Column('location:N', title='Location'),
                columns=3  # Adjust number of columns as needed
            ).resolve_scale(
                x='independent',
                y='independent'
            )
            
            st.altair_chart(scatter_chart, use_container_width=True)
        
        # Create a combined visualization showing all three dimensions
        st.subheader("3D Parameter Space (Location-Log(Scale)-Trend)")
        
        # Use a more complex scatter plot with size encoding for the third dimension, faceted by location
        combined_chart = alt.Chart(params_df_clean).mark_circle(
            opacity=0.7, color='darkblue'
        ).encode(
            x=alt.X('location_param:Q', title='Location Parameter'),
            y=alt.Y('log_scale_param:Q', title='Log(Scale) Parameter'),
            size=alt.Size('trend_param:Q', title='Trend Parameter', scale=alt.Scale(range=[50, 400])),
            tooltip=['location:N', 'season_idx:N', 'location_param:Q', 'log_scale_param:Q', 'trend_param:Q']
        ).properties(
            title='Combined Parameter Space (Size = Trend Parameter)',
            width=250,
            height=200
        ).facet(
            column=alt.Column('location:N', title='Location'),
            columns=3
        ).resolve_scale(
            x='independent',
            y='independent'
        )
        
        st.altair_chart(combined_chart, use_container_width=True)
    
    st.success("Analysis complete!")

if __name__ == "__main__":
    main()
