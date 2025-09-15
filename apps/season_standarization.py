import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).parent.parent))
from per_season.yearly_pattern import create_data_arrays

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
    st.dataframe(df.head())
    
    # Prepare the DataFrame with additional columns
    df = df.copy()
    df['log1p'] = np.log1p(df['disease_cases'])
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

    for location_idx, location in enumerate(locations):
        st.subheader(f"Location: {location}")
        
        location_data = df[df['location'] == location].copy()
        
        # Create charts using the DataFrame structure
        col1, col2 = st.columns(2)
        
        with col1:
            # Normalized data chart
            norm_chart = alt.Chart(location_data).mark_line(
                point=True, opacity=0.7
            ).encode(
                x=alt.X('season:O', title='Season (Month from Minimum)', scale=alt.Scale(domain=list(range(12)))),
                y=alt.Y('log1p_normalized:Q', title='Normalized Disease Cases'),
                color=alt.Color('season_idx:N', legend=alt.Legend(title="Season Year")),
                tooltip=['season:O', 'log1p_normalized:Q', 'season_idx:N', 'time_period:N']
            ).properties(
                title=f'{location} - Normalized Data',
                width=400,
                height=300
            )
            
            st.altair_chart(norm_chart, use_container_width=True)
        
        with col2:
            # Fully normalized data chart
            full_norm_chart = alt.Chart(location_data).mark_line(
                opacity=0.7
            ).encode(
                x=alt.X('season:O', title='Season (Month from Minimum)', scale=alt.Scale(domain=list(range(12)))),
                y=alt.Y('fully_normalized:Q', title='Fully Normalized Cases'),
                color=alt.Color('season_idx:N', legend=alt.Legend(title="Season Year")),
                tooltip=['season:O', 'fully_normalized:Q', 'season_idx:N', 'time_period:N']
            ).properties(
                title=f'{location} - Fully Normalized',
                width=400,
                height=300
            )
            
            st.altair_chart(full_norm_chart, use_container_width=True)
        
        # Seasonal pattern with confidence bands
        pattern_stats = location_data.groupby('season').agg({
            'pattern_mean': 'first',
            'pattern_std': 'first'
        }).reset_index()
        
        # Create confidence band data
        pattern_stats['upper'] = pattern_stats['pattern_mean'] + pattern_stats['pattern_std']
        pattern_stats['lower'] = pattern_stats['pattern_mean'] - pattern_stats['pattern_std']
        
        # Confidence band
        band = alt.Chart(pattern_stats).mark_area(
            opacity=0.3,
            color='lightblue'
        ).encode(
            x=alt.X('season:O', title='Season (Month from Minimum)'),
            y=alt.Y('lower:Q', title='Pattern Value'),
            y2=alt.Y2('upper:Q')
        )
        
        # Mean line
        mean_line = alt.Chart(pattern_stats).mark_line(
            color='black', 
            strokeWidth=3
        ).encode(
            x=alt.X('season:O'),
            y=alt.Y('pattern_mean:Q'),
            tooltip=['season:O', 'pattern_mean:Q', 'pattern_std:Q']
        )
        
        # Upper and lower bounds
        upper_line = alt.Chart(pattern_stats).mark_line(
            strokeDash=[5, 5],
            color='gray',
            opacity=0.7
        ).encode(
            x=alt.X('season:O'),
            y=alt.Y('upper:Q')
        )
        
        lower_line = alt.Chart(pattern_stats).mark_line(
            strokeDash=[5, 5],
            color='gray',
            opacity=0.7
        ).encode(
            x=alt.X('season:O'),
            y=alt.Y('lower:Q')
        )
        
        seasonal_chart = (band + mean_line + upper_line + lower_line).properties(
            title=f'{location} - Seasonal Pattern (Mean ± Std)',
            width=600,
            height=350
        )
        
        st.altair_chart(seasonal_chart, use_container_width=True)
        
        if location_idx >= 4:  # Limit to first 5 locations for performance
            remaining = len(locations) - location_idx - 1
            if remaining > 0:
                st.info(f"Showing first 5 locations. {remaining} more locations available.")
            break
    
    # Create summary comparison charts
    st.subheader("Summary Statistics")
    
    # Get pattern statistics for all locations
    pattern_summary = df.groupby(['location', 'season']).agg({
        'pattern_mean': 'first',
        'pattern_std': 'first'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        means_chart = alt.Chart(pattern_summary).mark_line(
            opacity=0.8, strokeWidth=2
        ).encode(
            x=alt.X('season:O', title='Season (Month from Minimum)', scale=alt.Scale(domain=list(range(12)))),
            y=alt.Y('pattern_mean:Q', title='Mean Normalized Cases'),
            color=alt.Color('location:N', legend=alt.Legend(title="Location")),
            tooltip=['season:O', 'pattern_mean:Q', 'location:N']
        ).properties(
            title='Seasonal Means by Location',
            width=400,
            height=350
        )
        
        st.altair_chart(means_chart, use_container_width=True)
    
    with col2:
        stds_chart = alt.Chart(pattern_summary).mark_line(
            opacity=0.8, strokeWidth=2
        ).encode(
            x=alt.X('season:O', title='Season (Month from Minimum)', scale=alt.Scale(domain=list(range(12)))),
            y=alt.Y('pattern_std:Q', title='Std Dev Normalized Cases'),
            color=alt.Color('location:N', legend=alt.Legend(title="Location")),
            tooltip=['season:O', 'pattern_std:Q', 'location:N']
        ).properties(
            title='Seasonal Standard Deviations by Location',
            width=400,
            height=350
        )
        
        st.altair_chart(stds_chart, use_container_width=True)
    
    st.success("Analysis complete!")

if __name__ == "__main__":
    main()
