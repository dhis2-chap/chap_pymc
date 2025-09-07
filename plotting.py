import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

def plot_seasonal_effects(idata, locations, config: 'Config', output_file: str):
    """Plot seasonal effects with 90% confidence intervals for each location."""

    # Extract seasonal components from posterior - handle both centered and non-centered
    total_seasonal: np.ndarray = idata.posterior['total_seasonal'].values
    total_seasonal = total_seasonal.reshape((-1,)+total_seasonal.shape[-2:])
    #
    #
    #
    # if config.use_non_centered and "seasonal_base_raw_std" in idata.posterior.data_vars:
    #     # Non-centered parameterization
    #     seasonal_sigma_base = idata.posterior[
    #         "seasonal_sigma_base"
    #     ].values  # (chain, draw)
    #     seasonal_base_raw_std = idata.posterior[
    #         "seasonal_base_raw_std"
    #     ].values  # (chain, draw, seasonal_periods)
    #     seasonal_base_samples = (
    #         seasonal_base_raw_std * seasonal_sigma_base[:, :, np.newaxis]
    #     )
    #
    #     seasonal_sigma_loc = idata.posterior[
    #         "seasonal_sigma_loc"
    #     ].values  # (chain, draw)
    #     seasonal_loc_raw_std = idata.posterior[
    #         "seasonal_loc_raw_std"
    #     ].values  # (chain, draw, seasonal_periods, locations)
    #     seasonal_loc_samples = (
    #         seasonal_loc_raw_std * seasonal_sigma_loc[:, :, np.newaxis, np.newaxis]
    #     )
    # else:
    #     # Centered parameterization (fallback)
    #     seasonal_base_samples = idata.posterior[
    #         "seasonal_base_raw"
    #     ].values  # (chain, draw, seasonal_periods)
    #     seasonal_loc_samples = idata.posterior[
    #         "seasonal_loc_raw"
    #     ].values  # (chain, draw, seasonal_periods, locations)
    #
    # # Flatten chain and draw dimensions
    # n_chains, n_draws = seasonal_base_samples.shape[:2]
    # seasonal_base_flat = seasonal_base_samples.reshape(
    #     n_chains * n_draws, -1
    # )  # (samples, seasonal_periods)
    # seasonal_loc_flat = seasonal_loc_samples.reshape(
    #     n_chains * n_draws, config.seasonal_periods, len(locations)
    # )
    #
    # # Apply zero-sum constraints (as done in the model)
    # seasonal_base_centered = seasonal_base_flat - np.mean(
    #     seasonal_base_flat, axis=1, keepdims=True
    # )
    # seasonal_loc_centered = seasonal_loc_flat - np.mean(
    #     seasonal_loc_flat, axis=1, keepdims=True
    # )
    #
    # # Combine base + location-specific effects
    # total_seasonal = (
    #     seasonal_base_centered[:, :, np.newaxis] + seasonal_loc_centered
    # )  # (samples, periods, locations)
    #
    # # Calculate statistics
    seasonal_mean = np.mean(total_seasonal, axis=0)  # (periods, locations)
    seasonal_q05 = np.percentile(total_seasonal, 5, axis=0)  # 90% CI lower
    seasonal_q95 = np.percentile(total_seasonal, 95, axis=0)  # 90% CI upper

    # Create plot
    n_locs = len(locations)
    cols = min(3, n_locs)
    rows = (n_locs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if n_locs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, "__len__") else [axes]
    else:
        axes = axes.flatten()

    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for i, location in enumerate(locations):
        if i < len(axes):
            ax = axes[i]

            x = np.arange(config.seasonal_periods)

            # Plot mean seasonal effect
            ax.plot(
                x,
                seasonal_mean[:, i],
                "o-",
                color="blue",
                linewidth=2,
                markersize=6,
                label="Mean Effect",
            )

            # Plot 90% confidence interval
            ax.fill_between(
                x,
                seasonal_q05[:, i],
                seasonal_q95[:, i],
                alpha=0.3,
                color="blue",
                label="90% CI",
            )

            ax.set_title(
                f"Seasonal Effect - {location}", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Month")
            ax.set_ylabel("Seasonal Effect")
            ax.set_xticks(x)
            ax.set_xticklabels(months[: config.seasonal_periods], rotation=45)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax.legend()

    # Hide unused subplots
    for j in range(n_locs, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Seasonal effects plot saved to: {output_file}")


def plot_rw_variance(idata, output_file: str):
    """Plot random walk variance posterior distribution."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Posterior distribution
    az.plot_posterior(idata.posterior["sigma_rw"], ax=ax1, textsize=12)
    ax1.set_title(
        "Random Walk Variance (σ_rw)\nPosterior Distribution",
        fontsize=12,
        fontweight="bold",
    )

    # Plot 2: Trace plot
    sigma_rw_samples = idata.posterior["sigma_rw"].values
    n_chains = sigma_rw_samples.shape[0]

    for chain in range(n_chains):
        ax2.plot(sigma_rw_samples[chain, :], alpha=0.7, label=f"Chain {chain + 1}")

    ax2.set_title(
        "Random Walk Variance (σ_rw)\nTrace Plot", fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Draw")
    ax2.set_ylabel("σ_rw")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Random walk variance plot saved to: {output_file}")


def plot_location_effects(idata, locations, output_file: str):
    """Plot IID location effects with uncertainty."""

    # Extract location effects from posterior
    location_samples = idata.posterior[
        "location_raw"
    ].values  # (chain, draw, locations)

    # Flatten chain and draw dimensions
    n_chains, n_draws = location_samples.shape[:2]
    location_flat = location_samples.reshape(
        n_chains * n_draws, -1
    )  # (samples, locations)

    # Apply zero-sum constraint (as done in the model)
    location_centered = location_flat# - np.mean(location_flat, axis=1, keepdims=True)

    # Calculate statistics
    location_mean = np.mean(location_centered, axis=0)
    location_q05 = np.percentile(location_centered, 5, axis=0)  # 90% CI lower
    location_q95 = np.percentile(location_centered, 95, axis=0)  # 90% CI upper

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Location effects with error bars
    x = np.arange(len(locations))
    ax1.errorbar(
        x,
        location_mean,
        yerr=[location_mean - location_q05, location_q95 - location_mean],
        fmt="o",
        capsize=5,
        capthick=2,
        markersize=8,
        linewidth=2,
    )

    ax1.set_title(
        "IID Location Effects\nwith 90% Confidence Intervals",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Location")
    ax1.set_ylabel("Location Effect")
    ax1.set_xticks(x)
    ax1.set_xticklabels(locations, rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # Plot 2: Posterior distributions for each location
    for i, location in enumerate(locations):
        ax2.hist(
            location_centered[:, i], bins=50, alpha=0.6, label=location, density=True
        )

    ax2.set_title(
        "Location Effects\nPosterior Distributions", fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Location Effect Value")
    ax2.set_ylabel("Density")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Location effects plot saved to: {output_file}")


def plot_all_parameter_samples(idata, config: 'Config', output_file: str):
    """Create a comprehensive figure showing sampling diagnostics for all parameters."""

    # Define parameter categories and their display properties
    param_info = {
        # Core parameters (always present)
        'intercept': {'label': 'Intercept', 'color': 'blue'},
        'sigma_rw': {'label': 'Random Walk σ', 'color': 'green'},
        'alpha': {'label': 'Overdispersion α', 'color': 'red'},

        # Covariate parameters (present if covariates exist)
        'beta': {'label': 'Covariate β', 'color': 'purple', 'multi_dim': True},

        # Seasonal parameters (present if seasonal effects enabled)
        'seasonal_sigma_base': {'label': 'Base Seasonal σ', 'color': 'orange'},
        'seasonal_sigma_loc': {'label': 'Location Seasonal σ', 'color': 'brown'},

        # Location parameters (present if location effects enabled)
        'location_sigma': {'label': 'Location Effects σ', 'color': 'pink'},
    }

    # Filter to only available parameters
    available_params = []
    for param_name, info in param_info.items():
        if param_name in idata.posterior.data_vars:
            available_params.append((param_name, info))

    n_params = len(available_params)
    if n_params == 0:
        print("No parameters found for plotting")
        return

    # Create figure with subplots - 2 columns (trace + posterior) per parameter
    cols = 2
    rows = n_params
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    # Handle case of single parameter
    if n_params == 1:
        axes = axes.reshape(1, -1)

    for i, (param_name, info) in enumerate(available_params):
        param_data = idata.posterior[param_name]

        # Handle multi-dimensional parameters
        if info.get('multi_dim', False) and param_data.ndim > 2:
            # For multi-dimensional parameters like beta, plot the first dimension
            if param_data.ndim == 3:  # (chain, draw, dim)
                param_values = param_data.values[:, :, 0]  # Take first coefficient
                param_label = f"{info['label']}[0]"
            else:
                param_values = param_data.values.reshape(param_data.shape[0], param_data.shape[1])
                param_label = info['label']
        else:
            param_values = param_data.values
            param_label = info['label']

        color = info['color']

        # Left subplot: Trace plot
        ax_trace = axes[i, 0]
        n_chains = param_values.shape[0]

        for chain in range(n_chains):
            ax_trace.plot(param_values[chain, :], alpha=0.8, color=color,
                          label=f'Chain {chain+1}' if i == 0 else "")

        ax_trace.set_title(f'{param_label} - Trace Plot', fontsize=10, fontweight='bold')
        ax_trace.set_xlabel('Draw')
        ax_trace.set_ylabel('Value')
        ax_trace.grid(True, alpha=0.3)
        if i == 0:  # Only show legend for first parameter
            ax_trace.legend()

        # Right subplot: Posterior distribution
        ax_posterior = axes[i, 1]

        # Flatten all chains for posterior
        param_flat = param_values.flatten()

        # Create histogram
        ax_posterior.hist(param_flat, bins=50, density=True, alpha=0.6, color=color, edgecolor='black')

        # Add summary statistics
        mean_val = np.mean(param_flat)
        median_val = np.median(param_flat)
        q025 = np.percentile(param_flat, 2.5)
        q975 = np.percentile(param_flat, 97.5)

        # Add vertical lines for summary stats
        ax_posterior.axvline(mean_val, color='red', linestyle='-', alpha=0.8, label='Mean')
        ax_posterior.axvline(median_val, color='black', linestyle='--', alpha=0.8, label='Median')
        ax_posterior.axvline(q025, color='gray', linestyle=':', alpha=0.6, label='95% CI')
        ax_posterior.axvline(q975, color='gray', linestyle=':', alpha=0.6)

        ax_posterior.set_title(f'{param_label} - Posterior Distribution', fontsize=10, fontweight='bold')
        ax_posterior.set_xlabel('Value')
        ax_posterior.set_ylabel('Density')
        ax_posterior.grid(True, alpha=0.3)
        if i == 0:  # Only show legend for first parameter
            ax_posterior.legend(fontsize=8)

        # Add text box with summary statistics
        stats_text = f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\n95% CI: [{q025:.3f}, {q975:.3f}]'
        ax_posterior.text(0.02, 0.98, stats_text, transform=ax_posterior.transAxes,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=8)

    plt.suptitle('Model Parameter Sampling Diagnostics', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Parameter samples plot saved to: {output_file}")


def plot_parameter_summary_table(idata, config: 'Config', output_file: str):
    """Create a summary table of all parameters with key statistics."""

    # Get all scalar parameters (exclude multi-dimensional random effects)
    scalar_params = []
    for var_name in idata.posterior.data_vars:
        var_data = idata.posterior[var_name]
        # Include only scalar parameters or first element of vector parameters
        if var_data.ndim <= 2:  # (chain, draw)
            scalar_params.append(var_name)
        elif var_data.ndim == 3 and var_name == 'beta':  # Special case for beta coefficients
            # Add each beta coefficient separately
            n_betas = var_data.shape[2]
            for j in range(n_betas):
                scalar_params.append(f'beta[{j}]')

    # Calculate summary statistics
    stats_data = []
    for param in scalar_params:
        if '[' in param:  # Handle indexed parameters like beta[0]
            base_param = param.split('[')[0]
            idx = int(param.split('[')[1].split(']')[0])
            values = idata.posterior[base_param].values[:, :, idx].flatten()
        else:
            values = idata.posterior[param].values.flatten()

        stats_data.append({
            'Parameter': param,
            'Mean': np.mean(values),
            'Std': np.std(values),
            'Median': np.median(values),
            '2.5%': np.percentile(values, 2.5),
            '97.5%': np.percentile(values, 97.5),
            'R-hat': float(az.rhat(idata.posterior[param.split('[')[0]] if '[' in param else idata.posterior[param]).values.flatten()[0]) if len(values) > 4 else np.nan
        })

    # Create table plot
    fig, ax = plt.subplots(figsize=(12, len(stats_data) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Convert to DataFrame for easier table creation
    import pandas as pd
    df_stats = pd.DataFrame(stats_data)

    # Format numbers for display
    for col in ['Mean', 'Std', 'Median', '2.5%', '97.5%']:
        df_stats[col] = df_stats[col].apply(lambda x: f'{x:.3f}')
    df_stats['R-hat'] = df_stats['R-hat'].apply(lambda x: f'{x:.3f}' if not np.isnan(x) else 'N/A')

    # Create table
    table = ax.table(cellText=df_stats.values, colLabels=df_stats.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Header styling
    for i in range(len(df_stats.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df_stats) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(df_stats.columns)):
            table[(i, j)].set_facecolor(color)

    plt.title('Model Parameter Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Parameter summary table saved to: {output_file}")

def create_model_visualization(train_data_file: str, historic_data_file: str,
                               predictions_file: str, output_file: str,
                               config: 'Config'):
    """Create comprehensive visualization of model training data, historic data, and predictions."""

    # Load all datasets
    train_df = pd.read_csv(train_data_file)
    historic_df = pd.read_csv(historic_data_file)
    pred_df = pd.read_csv(predictions_file)

    # Convert time periods to datetime
    train_df['time_period'] = pd.to_datetime(train_df['time_period'])
    historic_df['time_period'] = pd.to_datetime(historic_df['time_period'])
    pred_df['time_period'] = pd.to_datetime(pred_df['time_period'])

    # Get all locations
    locations = sorted(train_df['location'].unique())
    n_locs = len(locations)

    # Set up the plot
    fig, axes = plt.subplots(n_locs, 1, figsize=(15, 4 * n_locs))
    if n_locs == 1:
        axes = [axes]

    # Calculate prediction statistics
    sample_cols = [col for col in pred_df.columns if col.startswith('sample_')]
    pred_df['pred_mean'] = pred_df[sample_cols].mean(axis=1)
    pred_df['pred_std'] = pred_df[sample_cols].std(axis=1)
    pred_df['pred_lower'] = pred_df[sample_cols].quantile(0.025, axis=1)
    pred_df['pred_upper'] = pred_df[sample_cols].quantile(0.975, axis=1)

    for i, location in enumerate(locations):
        ax = axes[i]

        # Filter data for this location
        train_loc = train_df[train_df['location'] == location].copy()
        historic_loc = historic_df[historic_df['location'] == location].copy()
        pred_loc = pred_df[pred_df['location'] == location].copy()

        # Sort by time
        train_loc = train_loc.sort_values('time_period')
        historic_loc = historic_loc.sort_values('time_period')
        pred_loc = pred_loc.sort_values('time_period')

        # Plot training data
        ax.plot(train_loc['time_period'], train_loc['disease_cases'],
                'o-', color='blue', alpha=0.7, label='Training Data', markersize=4)

        # Plot historic data (if different from training)
        if len(historic_loc) > 0:
            # Only plot historic data that's not already in training
            train_periods = set(train_loc['time_period'])
            historic_new = historic_loc[~historic_loc['time_period'].isin(train_periods)]
            if len(historic_new) > 0:
                ax.plot(historic_new['time_period'], historic_new['disease_cases'],
                        's-', color='green', alpha=0.7, label='Historic Data', markersize=4)

        # Plot predictions with uncertainty
        if len(pred_loc) > 0:
            # Prediction mean
            ax.plot(pred_loc['time_period'], pred_loc['pred_mean'],
                    'D-', color='red', alpha=0.8, label='Predicted Mean', markersize=5)

            # Prediction uncertainty
            ax.fill_between(pred_loc['time_period'],
                            pred_loc['pred_lower'],
                            pred_loc['pred_upper'],
                            alpha=0.3, color='red', label='95% Prediction Interval')

        # Formatting
        ax.set_title(f'Disease Cases - {location}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Disease Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)

        # Set y-axis to start from 0 for better comparison
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_file}")


def create_parameter_plot(model_file: str, output_file: str, config: 'Config'):
    """Create visualization of model parameters from posterior."""

    # Load model data
    base_name = model_file.rsplit('.', 1)[0] if '.' in model_file else model_file
    idata_filename = f"{base_name}_idata.nc"

    try:
        idata = az.from_netcdf(idata_filename)
    except FileNotFoundError:
        print(f"Could not find inference data file: {idata_filename}")
        return

    # Create parameter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot 1: Random Walk variance
    if 'sigma_rw' in idata.posterior:
        az.plot_posterior(idata.posterior['sigma_rw'], ax=axes[0])
        axes[0].set_title('Random Walk Variance (σ_rw)')

    # Plot 2: Overdispersion parameter
    if 'alpha' in idata.posterior:
        az.plot_posterior(idata.posterior['alpha'], ax=axes[1])
        axes[1].set_title('Negative Binomial Overdispersion (α)')

    # Plot 3: Seasonal effects (if enabled)
    if config.use_seasonal_effects and 'seasonal_sigma_base' in idata.posterior:
        az.plot_posterior(idata.posterior['seasonal_sigma_base'], ax=axes[2])
        axes[2].set_title('Base Seasonal Effect Variance')
    else:
        axes[2].text(0.5, 0.5, 'Seasonal Effects\nDisabled',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Seasonal Effects')

    # Plot 4: Location effects (if enabled)
    if config.use_location_effects and 'location_sigma' in idata.posterior:
        az.plot_posterior(idata.posterior['location_sigma'], ax=axes[3])
        axes[3].set_title('Location Effect Variance')
    else:
        axes[3].text(0.5, 0.5, 'Location Effects\nDisabled',
                     ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Location Effects')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Parameter visualization saved to: {output_file}")


def plot_model_components(model_file: str, output_dir: str, config: 'Config'):
    """Create specialized plots of model components from posterior samples."""

    # Load model data
    base_name = model_file.rsplit('.', 1)[0] if '.' in model_file else model_file
    idata_filename = f"{base_name}_idata.nc"
    data_filename = f"{base_name}_data.pkl"

    try:
        idata = az.from_netcdf(idata_filename)
        with open(data_filename, 'rb') as f:
            data_dict, _ = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Could not find model files: {e}")
        return

    # Get location names
    locations = data_dict['locs']

    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot seasonal effects with 90% CI for each location
    #if config.use_seasonal_effects and 'seasonal_base_raw' in idata.posterior:
    plot_seasonal_effects(idata, locations, config, f"{output_dir}/seasonal_effects.png")

    # 2. Plot random walk variance posterior
    if 'sigma_rw' in idata.posterior:
        plot_rw_variance(idata, f"{output_dir}/rw_variance.png")

    # 3. Plot IID location effects (if enabled)
    if config.use_location_effects and 'location_raw' in idata.posterior:
        plot_location_effects(idata, locations, f"{output_dir}/location_effects.png")

    # 4. Create comprehensive parameter sampling figure
    plot_all_parameter_samples(idata, config, f"{output_dir}/parameter_samples.png")

    print(f"Model component plots saved to directory: {output_dir}")
