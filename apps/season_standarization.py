import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from per_season.yearly_pattern import create_data_arrays

alt.data_transformers.enable('json')

st.set_page_config(page_title="Season Standardization Analysis", layout="wide")

def main():
    st.title("Season Standardization Analysis")
    st.markdown("Interactive visualization of seasonal disease patterns and normalization")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Data loaded successfully! Shape: {df.shape}")
            
            if st.button("Generate Analysis"):
                with st.spinner("Processing data and generating plots..."):
                    analyze_data(df)
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis")
        
        if st.checkbox("Use sample data"):
            sample_path = "/Users/knutdr/Downloads/dataset2/training_data.csv"
            if Path(sample_path).exists():
                df = pd.read_csv(sample_path)
                st.success(f"Sample data loaded! Shape: {df.shape}")
                
                if st.button("Generate Analysis with Sample Data"):
                    with st.spinner("Processing data and generating plots..."):
                        analyze_data(df)
            else:
                st.warning("Sample data file not found")

def analyze_data(df: pd.DataFrame):
    st.subheader("Data Overview")
    st.dataframe(df.head())
    
    df['log1p'] = np.log1p(df['disease_cases'])
    month_index = df['time_period'].apply(lambda x: int(x.split('-')[1]))
    df['month'] = month_index
    means = ((month, group['disease_cases'].mean()) for month, group in df.groupby('month'))
    min_month, val = min(means, key=lambda x: x[1])
    
    st.subheader(f"Analysis Details")
    st.write(f"Minimum disease cases month: {min_month} (avg cases: {val:.2f})")
    
    horizon = st.slider("Forecast Horizon", min_value=1, max_value=12, value=3)
    
    all_means = []
    all_stds = []
    full_year_data = []
    
    locations = df['location'].unique()
    
    for location_idx, (location, group) in enumerate(df.groupby('location')):
        st.subheader(f"Location: {location}")
        
        group = group.copy()
        group['log1p'] = group['log1p'].interpolate()
        cutoff_month_index = np.flatnonzero(df['month'] == min_month)[0]
        extra_offset = (len(group)-cutoff_month_index+horizon)%12
        ds = group['log1p'].values
        
        if np.isnan(ds).any() or np.isinf(ds).any():
            st.warning(f"Data quality issues found in {location}")
            continue
            
        ds = np.append(ds, [np.nan]*horizon)
        normies = []
        year_data_per_loc = []
        extra_offset = extra_offset if extra_offset <= horizon else 0
        
        norm_chart_data = []
        
        for i in range(len(ds) // 12):
            year_data = ds[cutoff_month_index + i * 12:cutoff_month_index + (i+1) * 12+extra_offset]
            missing = 12+extra_offset-len(year_data)
            if missing > 0:
                year_data = np.append(year_data, [np.nan]*missing)
            year_data_per_loc.append(year_data)
            normalized = (year_data - year_data[:12].mean()) / max(year_data[:12].std(), 0.001)
            normies.append(normalized)
            
            for month_idx, value in enumerate(normalized):
                if not np.isnan(value):
                    norm_chart_data.append({
                        'Month': month_idx,
                        'Value': value,
                        'Year': f'Year {i+1}',
                        'Type': 'Normalized'
                    })
        
        year_data_per_loc = np.array(year_data_per_loc)
        full_year_data.append(year_data_per_loc)
        normies = np.array(normies)
        means = np.nanmean(normies, axis=0)
        stds = np.nanstd(normies, axis=0)
        
        if np.isnan(means).any() or np.isnan(stds).any():
            st.warning(f"Standardization issues found in {location}")
            continue
            
        all_means.append(means)
        all_stds.append(stds)
        fully_norm = (normies-means)/stds
        
        # Add fully normalized data to chart data
        for i, fn in enumerate(fully_norm):
            for month_idx, value in enumerate(fn):
                if not np.isnan(value):
                    norm_chart_data.append({
                        'Month': month_idx,
                        'Value': value,
                        'Year': f'Year {i+1}',
                        'Type': 'Fully Normalized'
                    })
        
        # Create Altair charts
        if norm_chart_data:
            chart_df = pd.DataFrame(norm_chart_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                norm_chart = alt.Chart(chart_df[chart_df['Type'] == 'Normalized']).mark_line(
                    point=True, opacity=0.7
                ).add_selection(
                    alt.selection_multi(fields=['Year'])
                ).encode(
                    x=alt.X('Month:O', title='Month'),
                    y=alt.Y('Value:Q', title='Normalized Disease Cases'),
                    color=alt.Color('Year:N', legend=alt.Legend(title="Year")),
                    tooltip=['Month:O', 'Value:Q', 'Year:N']
                ).properties(
                    title=f'{location} - Normalized Data',
                    width=400,
                    height=300
                ).resolve_scale(color='independent')
                
                st.altair_chart(norm_chart, use_container_width=True)
            
            with col2:
                full_norm_chart = alt.Chart(chart_df[chart_df['Type'] == 'Fully Normalized']).mark_line(
                    opacity=0.7
                ).add_selection(
                    alt.selection_multi(fields=['Year'])
                ).encode(
                    x=alt.X('Month:O', title='Month'),
                    y=alt.Y('Value:Q', title='Fully Normalized Cases'),
                    color=alt.Color('Year:N', legend=alt.Legend(title="Year")),
                    tooltip=['Month:O', 'Value:Q', 'Year:N']
                ).properties(
                    title=f'{location} - Fully Normalized',
                    width=400,
                    height=300
                ).resolve_scale(color='independent')
                
                st.altair_chart(full_norm_chart, use_container_width=True)
        
        # Create seasonal pattern chart data
        seasonal_data = []
        months = range(len(means))
        for month in months:
            seasonal_data.extend([
                {'Month': month, 'Value': means[month], 'Type': 'Mean'},
                {'Month': month, 'Value': means[month] + stds[month], 'Type': 'Upper'},
                {'Month': month, 'Value': means[month] - stds[month], 'Type': 'Lower'}
            ])
        
        seasonal_df = pd.DataFrame(seasonal_data)
        
        # Create confidence band
        band = alt.Chart(seasonal_df[seasonal_df['Type'].isin(['Upper', 'Lower'])]).mark_area(
            opacity=0.3,
            color='lightblue'
        ).encode(
            x=alt.X('Month:O'),
            y=alt.Y('Value:Q', scale=alt.Scale(zero=False)),
            y2='Value:Q'
        ).transform_pivot(
            'Type', 'Value', groupby=['Month']
        ).transform_calculate(
            y='datum.Lower',
            y2='datum.Upper'
        )
        
        # Mean line
        mean_line = alt.Chart(seasonal_df[seasonal_df['Type'] == 'Mean']).mark_line(
            color='black', 
            strokeWidth=3
        ).encode(
            x=alt.X('Month:O', title='Month'),
            y=alt.Y('Value:Q', title='Normalized Disease Cases', scale=alt.Scale(zero=False)),
            tooltip=['Month:O', 'Value:Q']
        )
        
        # Upper and lower bounds
        bounds = alt.Chart(seasonal_df[seasonal_df['Type'].isin(['Upper', 'Lower'])]).mark_line(
            strokeDash=[5, 5],
            color='black',
            opacity=0.7
        ).encode(
            x=alt.X('Month:O'),
            y=alt.Y('Value:Q'),
            color=alt.Color('Type:N', scale=alt.Scale(range=['gray', 'gray']), legend=None)
        )
        
        seasonal_chart = (band + mean_line + bounds).properties(
            title=f'{location} - Seasonal Pattern (Mean ± Std)',
            width=600,
            height=350
        ).resolve_scale(y='shared')
        
        st.altair_chart(seasonal_chart, use_container_width=True)
        
        if location_idx >= 4:  # Limit to first 5 locations for performance
            remaining = len(locations) - location_idx - 1
            if remaining > 0:
                st.info(f"Showing first 5 locations. {remaining} more locations available.")
            break
    
    if all_means and all_stds:
        st.subheader("Summary Statistics")
        all_means_arr = np.array(all_means)
        all_stds_arr = np.array(all_stds)
        
        # Create summary data for Altair
        summary_data = []
        for i, (means, stds, location) in enumerate(zip(all_means_arr, all_stds_arr, locations[:len(all_means_arr)])):
            for month, (mean_val, std_val) in enumerate(zip(means, stds)):
                summary_data.extend([
                    {'Month': month, 'Value': mean_val, 'Location': location, 'Metric': 'Mean'},
                    {'Month': month, 'Value': std_val, 'Location': location, 'Metric': 'Std Dev'}
                ])
        
        summary_df = pd.DataFrame(summary_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            means_chart = alt.Chart(summary_df[summary_df['Metric'] == 'Mean']).mark_line(
                opacity=0.8, strokeWidth=2
            ).encode(
                x=alt.X('Month:O', title='Month'),
                y=alt.Y('Value:Q', title='Mean Normalized Cases'),
                color=alt.Color('Location:N', legend=alt.Legend(title="Location")),
                tooltip=['Month:O', 'Value:Q', 'Location:N']
            ).properties(
                title='Seasonal Means by Location',
                width=400,
                height=350
            )
            
            st.altair_chart(means_chart, use_container_width=True)
        
        with col2:
            stds_chart = alt.Chart(summary_df[summary_df['Metric'] == 'Std Dev']).mark_line(
                opacity=0.8, strokeWidth=2
            ).encode(
                x=alt.X('Month:O', title='Month'),
                y=alt.Y('Value:Q', title='Std Dev Normalized Cases'),
                color=alt.Color('Location:N', legend=alt.Legend(title="Location")),
                tooltip=['Month:O', 'Value:Q', 'Location:N']
            ).properties(
                title='Seasonal Standard Deviations by Location',
                width=400,
                height=350
            )
            
            st.altair_chart(stds_chart, use_container_width=True)
        
        st.success("Analysis complete!")
    else:
        st.error("No valid data processed. Please check your data format.")

if __name__ == "__main__":
    main()
