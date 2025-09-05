# pymc_st_arf_model_nomutable.py
# Temporal PyMC model with Random Walk in time and independent location effects.
# No pm.MutableData / pm.Data usage — just plain NumPy constants.
#
# Usage:
#   python example.py train training_data.csv model.pkl config.yaml
#   python example.py predict model.pkl historic_data.csv future_data.csv predictions.csv config.yaml
#   python example.py plot model.pkl training_data.csv historic_data.csv predictions.csv config.yaml output.png
#   python example.py plot-components model.pkl config.yaml output_dir

import argparse
import uuid
import pickle
import json
from _ast import arg

import pytest
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pydantic
import pymc as pm
import pytensor.tensor as pt
import cyclopts
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

class Config(pydantic.BaseModel):
    covariate_names: list[str] = ['rainfall', 'mean_temperature']
    horizon: int = 3
    tune: int = 100
    draws: int = 100
    chains: int = 2
    seed: int = 42
    freq: str = "MS"  # Monthly start frequency
    target_col: str = 'disease_cases'
    date_col: str = 'time_period'
    loc_col: str = 'location'
    # Seasonal random effect parameters
    use_seasonal_effects: bool = True
    seasonal_periods: int = 12  # 12 months in a year
    seasonal_sigma_base: float = 1.0  # Prior std for base seasonal effect
    seasonal_sigma_loc: float = 0.5   # Prior std for location-specific seasonal effects
    # Location random effect parameters
    use_location_effects: bool = True
    location_sigma: float = 1.0  # Prior std for location random effects

def complete_monthly_panel(df, date_col, loc_col, cols, freq="MS"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([loc_col, date_col])
    all_locs = df[loc_col].unique().tolist()
    tmin, tmax = df[date_col].min(), df[date_col].max()
    full_time = pd.date_range(tmin, tmax, freq=freq)
    chunks = []
    for loc in all_locs:
        g = df[df[loc_col] == loc].set_index(date_col).reindex(full_time)
        g[loc_col] = loc
        chunks.append(g[cols + [loc_col]])
    out = pd.concat(chunks).reset_index().rename(columns={"index": date_col})
    return out, full_time, all_locs

def to_tensor_panels(pnl, time_idx, locs, date_col, loc_col, col):
    T, L = len(time_idx), len(locs)
    pivot = pnl.pivot(index=date_col, columns=loc_col, values=col).reindex(time_idx)
    M = np.full((T, L), np.nan, dtype=float)
    for j, loc in enumerate(locs):
        M[:, j] = pivot[loc].to_numpy()
    return M

def safe_impute(M):
    df = pd.DataFrame(M)
    df = df.interpolate(limit_direction="both").ffill().bfill()
    return df.to_numpy()

def extract_month_indices(time_idx):
    """Extract month indices (0-11) from time index for seasonal effects."""
    months = pd.to_datetime(time_idx).month - 1  # Convert to 0-11
    return months.values

def prepare_data(args: Config, raw):
    """Prepare and preprocess data for training."""

    # Add population to required columns if available
    available_cols = raw.columns.tolist()
    keep_cols = [args.target_col] + args.covariate_names
    has_population = 'population' in available_cols
    if has_population:
        keep_cols.append('population')
    else:
        print("Warning: Population column not found. Model will run without population offset.")
    panel, time_idx, locs = complete_monthly_panel(raw, args.date_col, args.loc_col, keep_cols, freq=args.freq)
    T, L, H = len(time_idx), len(locs), args.horizon

    # Target (T, L)
    y = to_tensor_panels(panel, time_idx, locs, 'time_period', 'location', 'disease_cases')
    # if np.isnan(y).any():
    #    y = safe_impute(y)
    y = np.clip(y, 0, None).astype(int)
    
    # Population (T, L) for offset - only if available
    if has_population:
        pop = to_tensor_panels(panel, time_idx, locs, 'time_period', 'location', 'population')

        # Forward fill any missing population values
        if np.isnan(pop).any():
            pop = safe_impute(pop)
        log_pop_offset = np.log(np.clip(pop, 1.0, None))  # Ensure positive population
    else:
        # Create zero offset if no population data (equivalent to no offset)
        log_pop_offset = np.zeros((T, L))

    # Covariates (optional) -> standardize
    feat_names = args.covariate_names
    X_list = [to_tensor_panels(panel, time_idx, locs, args.date_col, args.loc_col, col) for col in feat_names]
    X = np.stack(X_list, axis=2)  # (T, L, P)
    P = X.shape[2]
    scaler = StandardScaler().fit(X.reshape(T*L, P))
    X_std = scaler.transform(X.reshape(T*L, P)).reshape(T, L, P)
    
    # Extract month indices for seasonal effects
    month_indices = extract_month_indices(time_idx) if args.use_seasonal_effects else None
    
    return {
        'y': y,
        'X_std': X_std,
        'scaler': scaler,
        'time_idx': time_idx,
        'locs': locs,
        'feat_names': feat_names,
        'T': T,
        'L': L,
        'P': P,
        'H': H,
        'month_indices': month_indices,
        'log_pop_offset': log_pop_offset
    }


def format_predictions(ppc, data_dict, args):
    """Format predictions into output DataFrame."""
    time_idx = data_dict['time_idx']
    locs = data_dict['locs']
    H = data_dict['H']
    
    # Generate future dates
    fut_dates = pd.date_range(time_idx[-1] + pd.tseries.frequencies.to_offset(args.freq),
                              periods=H, freq=args.freq)
    
    # Find the y_fut variable (it has a unique ID suffix)
    y_fut_var = None
    for var_name in ppc.prior.data_vars:
        if var_name.startswith('y_fut_'):
            y_fut_var = var_name
            break
    
    if y_fut_var is None:
        raise ValueError("Could not find y_fut variable in predictions")
    
    fut_Y = ppc.prior[y_fut_var]  # (chain, draws, H, L)
    # Flatten chain and draws dimensions  
    fut_Y = fut_Y.stack(sample=('chain', 'draw'))  # (sample, H, L)
    fut_dates = [str(p)[:7] for p in fut_dates]
    
    rows = [
        [fut_dates[h], locs[l]] + list(fut_Y[h, l, :].values.tolist())
        for h in range(H)
        for l in range(len(locs))
    ]
    col_names = ['time_period', 'location'] + [f'sample_{i}' for i in range(fut_Y.shape[-1])]
    out_df = pd.DataFrame(rows, columns=col_names)
    out_df.to_csv('forecast_samples.csv', index=False)
    return out_df

def prepare_extended_data(data_dict, historic_data, args):
    """Prepare extended data that includes historic observations beyond training period."""
    # Get original training data info
    original_time_idx = data_dict['time_idx']
    original_locs = data_dict['locs']
    scaler = data_dict['scaler']
    T_orig = data_dict['T']
    L = data_dict['L'] 
    P = data_dict['P']
    
    # Find the last training date
    last_training_date = original_time_idx[-1]
    
    # Filter historic data to only include periods after training
    historic_data = historic_data.copy()
    historic_data[args.date_col] = pd.to_datetime(historic_data[args.date_col])
    extended_historic = historic_data[historic_data[args.date_col] > last_training_date]
    
    if len(extended_historic) == 0:
        # No new historic data, return original data
        return {
            **data_dict,
            'T_extended': T_orig,
            'X_extended': data_dict['X_std'],
            'extended_time_idx': original_time_idx,
            'extended_month_indices': data_dict['month_indices'],
            'log_pop_offset_extended': data_dict['log_pop_offset']
        }
    
    # Create extended time index
    extended_periods = extended_historic[args.date_col].unique()
    extended_periods = pd.to_datetime(extended_periods)
    extended_periods = pd.DatetimeIndex(extended_periods).sort_values()
    extended_time_idx = original_time_idx.union(extended_periods).sort_values()
    
    # Process extended historic data similar to training data
    available_cols = extended_historic.columns.tolist()
    keep_cols = [args.target_col] + args.covariate_names
    has_population = 'population' in available_cols
    if has_population:
        keep_cols.append('population')
    extended_panel, _, extended_locs = complete_monthly_panel(
        extended_historic, args.date_col, args.loc_col, keep_cols, freq=args.freq
    )
    
    # Ensure locations match original training data
    if set(extended_locs) != set(original_locs):
        print(f"Warning: Location mismatch. Original: {set(original_locs)}, Extended: {set(extended_locs)}")
    
    # Extract extended covariates and population for the new period only
    extended_start_idx = len(original_time_idx) 
    extended_end_idx = len(extended_time_idx)
    
    if P > 0:
        # Get covariate data for the extended period
        X_extended_new = []
        for col in args.covariate_names:
            X_col = to_tensor_panels(extended_panel, extended_time_idx, original_locs, 
                                   args.date_col, args.loc_col, col)
            X_extended_new.append(X_col[extended_start_idx:extended_end_idx, :])
        
        if X_extended_new:
            X_extended_new = np.stack(X_extended_new, axis=2)  # (T_new, L, P)
            # Standardize using original scaler
            T_new = X_extended_new.shape[0]
            X_extended_new_std = scaler.transform(X_extended_new.reshape(T_new*L, P)).reshape(T_new, L, P)
            # Combine with original standardized data
            X_extended = np.concatenate([data_dict['X_std'], X_extended_new_std], axis=0)
        else:
            X_extended = data_dict['X_std']
    else:
        X_extended = None
    
    # Handle population data for the extended period
    if has_population and extended_start_idx < extended_end_idx:
        # Get population data for the extended period
        pop_extended = to_tensor_panels(extended_panel, extended_time_idx, original_locs, 
                                      args.date_col, args.loc_col, 'population')
        # Forward fill any missing population values
        if np.isnan(pop_extended).any():
            pop_extended = safe_impute(pop_extended)
        log_pop_offset_extended = np.log(np.clip(pop_extended, 1.0, None))
    elif has_population:
        # No extension, use original population data
        log_pop_offset_extended = data_dict['log_pop_offset']
    else:
        # No population data available - create zero offset for extended period
        log_pop_offset_extended = np.zeros((len(extended_time_idx), L))
    
    # Extract month indices for seasonal effects (extended period)
    extended_month_indices = extract_month_indices(extended_time_idx) if args.use_seasonal_effects else None
    
    return {
        **data_dict,
        'T_extended': len(extended_time_idx),
        'X_extended': X_extended,
        'extended_time_idx': extended_time_idx,
        'extended_month_indices': extended_month_indices,
        'log_pop_offset_extended': log_pop_offset_extended
    }

def continue_rw_process(idata, T_orig, T_extended, sigma_rw_val, L):
    """Continue Random Walk process from training end through historic period."""
    if T_extended <= T_orig:
        # No extension needed
        u_rw_last = (
            idata.posterior["u_rw"]
            .isel(u_rw_dim_0=-1)
            .mean(dim=["chain", "draw"])
            .values
        )
        return pt.as_tensor_variable(u_rw_last)
    
    # Get starting point from training
    u_rw_last = (
        idata.posterior["u_rw"]
        .isel(u_rw_dim_0=-1)
        .mean(dim=["chain", "draw"])
        .values
    )
    
    # Continue Random Walk for the extended period
    steps_to_continue = T_extended - T_orig
    u_current = pt.as_tensor_variable(u_rw_last)
    
    # Generate Random Walk steps for the historic extension period
    # Random walk: u_t = u_{t-1} + noise_t
    for step in range(steps_to_continue):
        unique_id = str(uuid.uuid4())[:8]
        noise = pm.Normal(f"historic_noise_{unique_id}_{step}", 0.0, sigma_rw_val, shape=(L,))
        u_current = u_current + noise
    
    return u_current

def format_predictions_from_historic_end(ppc, extended_data_dict, args: Config):
    """Format predictions starting from end of historic data."""
    extended_time_idx = extended_data_dict['extended_time_idx']
    locs = extended_data_dict['locs']
    H = extended_data_dict['H']
    
    # Generate future dates starting from end of historic data
    fut_dates = pd.date_range(
        extended_time_idx[-1] + pd.tseries.frequencies.to_offset(args.freq),
        periods=H, freq=args.freq
    )
    
    # Find the y_fut variable (it has a unique ID suffix)
    y_fut_var = None
    for var_name in ppc.prior.data_vars:
        if var_name.startswith('y_fut_'):
            y_fut_var = var_name
            break
    
    if y_fut_var is None:
        raise ValueError("Could not find y_fut variable in predictions")
    
    fut_Y = ppc.prior[y_fut_var]  # (chain, draws, H, L)
    # Flatten chain and draws dimensions
    print(fut_Y.shape)
    fut_Y = fut_Y.stack(sample=('chain', 'draw'))  # (sample, H, L)
    print(fut_Y.shape)
    fut_dates = [str(p)[:7] for p in fut_dates]
    
    rows = [
        [fut_dates[h], locs[l]] + list(fut_Y[h, l, :].values.tolist())
        for h in range(H)
        for l in range(len(locs))
    ]

    sample_names = [f'sample_{i}' for i in range(fut_Y.shape[-1])]
    assert len(sample_names) == args.draws*args.chains, f"Sample count mismatch: expected {args.draws*args.chains}, got {len(sample_names)}"
    col_names = ['time_period', 'location'] + sample_names
    out_df = pd.DataFrame(rows, columns=col_names)
    
    return out_df

def on_train(training_data: pd.DataFrame, args: Config= Config()) -> tuple:
    '''This should train a model and return everything needed for prediction'''
    # We need to extract args from somewhere - for now, use default Args
    # In a real implementation, you might want to pass args differently

    # Prepare data directly
    # Prepare data and train model
    data_dict = prepare_data(args, training_data)
    """Train the PyMC model and return fitted model with inference data."""
    y = data_dict["y"]
    X_std = data_dict["X_std"]
    log_pop_offset = data_dict["log_pop_offset"]
    T, L, P, H = data_dict["T"], data_dict["L"], data_dict["P"], data_dict["H"]

    # Convert constants to graph tensors
    y_const = y
    Xpast_const = X_std if P else None
    log_pop_offset_const = log_pop_offset
    month_indices = data_dict["month_indices"]

    with pm.Model() as model:
        # ----- Fixed effects -----
        intercept = pm.Normal("intercept", 0.0, 5.0)
        if P:
            beta = pm.Normal("beta", 0.0, 1.0, shape=P)
            linpast = intercept + pt.tensordot(
                pt.as_tensor_variable(Xpast_const), beta, axes=[2, 0]
            )
        else:
            linpast = intercept + pt.zeros((T, L))

        # ----- Seasonal effects (if enabled) -----
        seasonal_effects = pt.zeros((T, L))
        if args.use_seasonal_effects:

            # Base seasonal effect with cyclical random walk
            seasonal_sigma_base = pm.HalfNormal("seasonal_sigma_base",
                                                args.seasonal_sigma_base)
            
            # Create cyclical random walk for base seasonal effect (12 months)
            # Use a random walk with wrap-around constraint
            seasonal_base_raw = pm.GaussianRandomWalk("seasonal_base_raw", 
                                                     sigma=seasonal_sigma_base, 
                                                     shape=args.seasonal_periods)
            # Apply zero-sum constraint to make it identifiable
            seasonal_base = seasonal_base_raw - pt.mean(seasonal_base_raw)
            
            # Location-specific seasonal effects (more regularized)
            seasonal_sigma_loc = pm.HalfNormal("seasonal_sigma_loc", args.seasonal_sigma_loc)
            seasonal_loc_raw = pm.Normal("seasonal_loc_raw", 0.0, seasonal_sigma_loc, 
                                       shape=(args.seasonal_periods, L))
            # Apply zero-sum constraint across months for each location
            seasonal_loc = seasonal_loc_raw - pt.mean(seasonal_loc_raw, axis=0)
            
            # Map seasonal effects to time periods using month indices
            month_tensor = pt.as_tensor_variable(month_indices)
            seasonal_base_mapped = seasonal_base[month_tensor]  # (T,)
            seasonal_loc_mapped = seasonal_loc[month_tensor, :]  # (T, L)
            
            # Total seasonal effect = base + location-specific
            seasonal_effects = seasonal_base_mapped[:, None] + seasonal_loc_mapped

        # ----- Location random effects (IID) -----
        location_effects = pt.zeros((T, L))
        if args.use_location_effects:
            # IID location random effects - constant across time for each location
            location_sigma = pm.HalfNormal("location_sigma", args.location_sigma)
            location_raw = pm.Normal("location_raw", 0.0, location_sigma, shape=L)
            # Apply zero-sum constraint to make it identifiable
            location_centered = location_raw - pt.mean(location_raw)
            # Broadcast to (T, L) - same effect for each location across all time periods
            location_effects = pt.tile(location_centered[None, :], (T, 1))

        # ----- Random Walk process for each location -----
        sigma_rw = pm.HalfNormal("sigma_rw", 1.0)

        # Random walk process for each location independently
        # GaussianRandomWalk creates a random walk along the time dimension
        u_seq = pm.GaussianRandomWalk("u_rw",
                                      sigma=sigma_rw,
                                      shape=(T, L))

        # ----- Observation model: Negative Binomial -----
        alpha = pm.HalfNormal("alpha", 1.0)
        # Add population offset to the linear predictor before exponentiating
        log_mu_past = linpast + seasonal_effects + location_effects + u_seq + pt.as_tensor_variable(log_pop_offset_const)
        mu_past = pt.exp(log_mu_past)
        y_like = pm.NegativeBinomial("y", mu=mu_past, alpha=alpha, observed=y_const)

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=0.9,
            random_seed=args.seed,
            progressbar=True,
        )
    # Return everything needed for prediction
    return (model, idata, data_dict, args)

def on_predict(model_and_data: tuple, historic_data: pd.DataFrame, args: Config= Config()) -> pd.DataFrame:
    '''Use trained model parameters to predic t from the end of historic_data'''
    # Unpack the training results
    model, idata, data_dict, args = model_and_data
    
    # Prepare extended data that includes historic observations
    extended_data_dict = prepare_extended_data(data_dict, historic_data, args)
    
    # Get dimensions
    T_orig, L, P, H = data_dict["T"], data_dict["L"], data_dict["P"], data_dict["H"]
    T_extended = extended_data_dict["T_extended"]
    
    # Prepare future covariates by holding last value from historic data
    X_extended = extended_data_dict["X_extended"]
    log_pop_offset_extended = extended_data_dict["log_pop_offset_extended"]
    
    if P > 0:
        X_future = np.tile(X_extended[-1:, :, :], (H, 1, 1))  # hold last from historic data
        Xfut_const = X_future
    else:
        Xfut_const = None

    # Prepare future population offset by holding last value from historic data
    log_pop_offset_future = np.tile(log_pop_offset_extended[-1:, :], (H, 1))  # (H, L)

    # Create a new model for prediction to avoid variable name conflicts
    with pm.Model() as pred_model:
        # Get parameter values from posterior
        intercept_val = idata.posterior["intercept"].mean(dim=["chain", "draw"]).values
        sigma_rw_val = idata.posterior["sigma_rw"].mean(dim=["chain", "draw"]).values
        alpha_val = idata.posterior["alpha"].mean(dim=["chain", "draw"]).values

        # Fixed effects for future periods
        if P:
            beta_val = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
            linfut = intercept_val + pt.tensordot(
                pt.as_tensor_variable(Xfut_const),
                pt.as_tensor_variable(beta_val),
                axes=[2, 0],
            )
        else:
            linfut = intercept_val + pt.zeros((H, L))

        # Seasonal effects for future periods
        seasonal_effects_fut = pt.zeros((H, L))
        if args.use_seasonal_effects:
            # Get seasonal parameters from posterior
            seasonal_base_val = idata.posterior["seasonal_base_raw"].mean(dim=["chain", "draw"]).values
            seasonal_base_val = seasonal_base_val - np.mean(seasonal_base_val)  # Apply zero-sum constraint
            
            seasonal_loc_val = idata.posterior["seasonal_loc_raw"].mean(dim=["chain", "draw"]).values
            seasonal_loc_val = seasonal_loc_val - np.mean(seasonal_loc_val, axis=0)  # Zero-sum constraint
            
            # Generate future month indices starting from end of historic data
            extended_time_idx = extended_data_dict['extended_time_idx']
            last_date = pd.Timestamp(extended_time_idx[-1])
            future_start_date = last_date + pd.tseries.frequencies.to_offset(args.freq)
            future_dates = pd.date_range(future_start_date, periods=H, freq=args.freq)
            future_month_indices = extract_month_indices(future_dates)
            
            # Map seasonal effects to future periods
            seasonal_base_fut = seasonal_base_val[future_month_indices]  # (H,)
            seasonal_loc_fut = seasonal_loc_val[future_month_indices, :]  # (H, L)
            
            # Total seasonal effect = base + location-specific
            seasonal_effects_fut = pt.as_tensor_variable(seasonal_base_fut)[:, None] + pt.as_tensor_variable(seasonal_loc_fut)

        # Location effects for future periods
        location_effects_fut = pt.zeros((H, L))
        if args.use_location_effects:
            # Get location parameters from posterior
            location_val = idata.posterior["location_raw"].mean(dim=["chain", "draw"]).values
            location_val = location_val - np.mean(location_val)  # Apply zero-sum constraint
            
            # Location effects are constant across time, so broadcast to (H, L)
            location_effects_fut = pt.as_tensor_variable(np.tile(location_val[None, :], (H, 1)))

        # Continue Random Walk from training through historic period to get final state
        u_rw_extended = continue_rw_process(idata, T_orig, T_extended, sigma_rw_val, L)

        # Create future Random Walk starting from end of historic period
        unique_id = str(uuid.uuid4())[:8]

        # For prediction, we'll use a simple Random Walk continuation
        # Initialize with last state after processing historic data
        u_fut = [u_rw_extended]

        # Generate H future steps
        for h in range(H):
            noise = pm.Normal(f"noise_{unique_id}_{h}", 0.0, sigma_rw_val, shape=(L,))
            next_u = u_fut[-1] + noise  # Random walk: u_t = u_{t-1} + noise_t
            u_fut.append(next_u)

        # Stack future Random Walk values (skip initial value)
        u_fut_seq = pt.stack(u_fut[1:], axis=0)  # (H, L)

        # Add population offset to future predictions
        log_mu_future = linfut + seasonal_effects_fut + location_effects_fut + u_fut_seq + pt.as_tensor_variable(log_pop_offset_future)
        mu_future = pt.exp(log_mu_future)
        y_fut = pm.NegativeBinomial(f"y_fut_{unique_id}", mu=mu_future, alpha=alpha_val)

        # Sample from the prediction model
        ppc = pm.sample_prior_predictive(
            samples=len(idata.posterior.chain) * len(idata.posterior.draw)
        )

    # Format predictions starting from end of historic data
    result_df = format_predictions_from_historic_end(ppc, extended_data_dict, args)
    
    return result_df


def create_model_visualization(train_data_file: str, historic_data_file: str, 
                             predictions_file: str, output_file: str, 
                             config: Config):
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


def create_parameter_plot(model_file: str, output_file: str, config: Config):
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


def plot_model_components(model_file: str, output_dir: str, config: Config):
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
    if config.use_seasonal_effects and 'seasonal_base_raw' in idata.posterior:
        plot_seasonal_effects(idata, locations, config, f"{output_dir}/seasonal_effects.png")
    
    # 2. Plot random walk variance posterior
    if 'sigma_rw' in idata.posterior:
        plot_rw_variance(idata, f"{output_dir}/rw_variance.png")
    
    # 3. Plot IID location effects (if enabled)
    if config.use_location_effects and 'location_raw' in idata.posterior:
        plot_location_effects(idata, locations, f"{output_dir}/location_effects.png")
    
    print(f"Model component plots saved to directory: {output_dir}")


def plot_seasonal_effects(idata, locations, config: Config, output_file: str):
    """Plot seasonal effects with 90% confidence intervals for each location."""
    
    # Extract seasonal components from posterior
    seasonal_base_samples = idata.posterior['seasonal_base_raw'].values  # (chain, draw, seasonal_periods)
    seasonal_loc_samples = idata.posterior['seasonal_loc_raw'].values   # (chain, draw, seasonal_periods, locations)
    
    # Flatten chain and draw dimensions
    n_chains, n_draws = seasonal_base_samples.shape[:2]
    seasonal_base_flat = seasonal_base_samples.reshape(n_chains * n_draws, -1)  # (samples, seasonal_periods)
    seasonal_loc_flat = seasonal_loc_samples.reshape(n_chains * n_draws, config.seasonal_periods, len(locations))
    
    # Apply zero-sum constraints (as done in the model)
    seasonal_base_centered = seasonal_base_flat - np.mean(seasonal_base_flat, axis=1, keepdims=True)
    seasonal_loc_centered = seasonal_loc_flat - np.mean(seasonal_loc_flat, axis=1, keepdims=True)
    
    # Combine base + location-specific effects
    total_seasonal = seasonal_base_centered[:, :, np.newaxis] + seasonal_loc_centered  # (samples, periods, locations)
    
    # Calculate statistics
    seasonal_mean = np.mean(total_seasonal, axis=0)  # (periods, locations)
    seasonal_q05 = np.percentile(total_seasonal, 5, axis=0)   # 90% CI lower
    seasonal_q95 = np.percentile(total_seasonal, 95, axis=0)  # 90% CI upper
    
    # Create plot
    n_locs = len(locations)
    cols = min(3, n_locs)
    rows = (n_locs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if n_locs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, location in enumerate(locations):
        if i < len(axes):
            ax = axes[i]
            
            x = np.arange(config.seasonal_periods)
            
            # Plot mean seasonal effect
            ax.plot(x, seasonal_mean[:, i], 'o-', color='blue', linewidth=2, 
                   markersize=6, label='Mean Effect')
            
            # Plot 90% confidence interval
            ax.fill_between(x, seasonal_q05[:, i], seasonal_q95[:, i], 
                           alpha=0.3, color='blue', label='90% CI')
            
            ax.set_title(f'Seasonal Effect - {location}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Seasonal Effect')
            ax.set_xticks(x)
            ax.set_xticklabels(months[:config.seasonal_periods], rotation=45)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.legend()
    
    # Hide unused subplots
    for j in range(n_locs, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Seasonal effects plot saved to: {output_file}")


def plot_rw_variance(idata, output_file: str):
    """Plot random walk variance posterior distribution."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Posterior distribution
    az.plot_posterior(idata.posterior['sigma_rw'], ax=ax1, textsize=12)
    ax1.set_title('Random Walk Variance (σ_rw)\nPosterior Distribution', fontsize=12, fontweight='bold')
    
    # Plot 2: Trace plot
    sigma_rw_samples = idata.posterior['sigma_rw'].values
    n_chains = sigma_rw_samples.shape[0]
    
    for chain in range(n_chains):
        ax2.plot(sigma_rw_samples[chain, :], alpha=0.7, label=f'Chain {chain+1}')
    
    ax2.set_title('Random Walk Variance (σ_rw)\nTrace Plot', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Draw')
    ax2.set_ylabel('σ_rw')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Random walk variance plot saved to: {output_file}")


def plot_location_effects(idata, locations, output_file: str):
    """Plot IID location effects with uncertainty."""
    
    # Extract location effects from posterior
    location_samples = idata.posterior['location_raw'].values  # (chain, draw, locations)
    
    # Flatten chain and draw dimensions
    n_chains, n_draws = location_samples.shape[:2]
    location_flat = location_samples.reshape(n_chains * n_draws, -1)  # (samples, locations)
    
    # Apply zero-sum constraint (as done in the model)
    location_centered = location_flat - np.mean(location_flat, axis=1, keepdims=True)
    
    # Calculate statistics
    location_mean = np.mean(location_centered, axis=0)
    location_q05 = np.percentile(location_centered, 5, axis=0)   # 90% CI lower
    location_q95 = np.percentile(location_centered, 95, axis=0)  # 90% CI upper
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Location effects with error bars
    x = np.arange(len(locations))
    ax1.errorbar(x, location_mean, 
                yerr=[location_mean - location_q05, location_q95 - location_mean],
                fmt='o', capsize=5, capthick=2, markersize=8, linewidth=2)
    
    ax1.set_title('IID Location Effects\nwith 90% Confidence Intervals', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Location')
    ax1.set_ylabel('Location Effect')
    ax1.set_xticks(x)
    ax1.set_xticklabels(locations, rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Posterior distributions for each location
    for i, location in enumerate(locations):
        ax2.hist(location_centered[:, i], bins=50, alpha=0.6, 
                label=location, density=True)
    
    ax2.set_title('Location Effects\nPosterior Distributions', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Location Effect Value')
    ax2.set_ylabel('Density')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Location effects plot saved to: {output_file}")


app = cyclopts.App()

@app.command()
def train(train_data: str, model: str, model_config: str):

    config = load_config(model_config)
    df = pd.read_csv(train_data)
    model_and_data = on_train(df, config)
    
    # Extract components that can be serialized
    model_dat, idata, data_dict, args = model_and_data
    
    # Create base filename without extension
    base_name = model.rsplit('.', 1)[0] if '.' in model else model
    
    # Save inference data using arviz (NetCDF format)
    idata_filename = f"{base_name}_idata.nc"
    idata.to_netcdf(idata_filename)
    
    # Save data dictionary and config using pickle
    data_filename = f"{base_name}_data.pkl"
    with open(data_filename, 'wb') as f:
        pickle.dump((data_dict, args), f)
    
    print(f"Model saved to:")
    print(f"  Inference data: {idata_filename}")
    print(f"  Data & config: {data_filename}")


def load_config(config_filename):
    with open(config_filename, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)



@app.command()
def predict(model: str,
            historic_data: str,
            future_data: str,
            out_file: str,
            model_config: str | None = None):
    # Create base filename without extension
    base_name = model.rsplit('.', 1)[0] if '.' in model else model
    args = load_config(model_config)
    # Load inference data from NetCDF
    idata_filename = f"{base_name}_idata.nc"
    idata = az.from_netcdf(idata_filename)
    
    # Load data dictionary and config from pickle
    data_filename = f"{base_name}_data.pkl"
    with open(data_filename, 'rb') as f:
        data_dict, args = pickle.load(f)
    
    # Create a dummy model object (we don't need the actual model for prediction)
    model = None
    
    # Reconstruct the model_and_data tuple
    model_and_data = (model, idata, data_dict, args)
    
    # Load historic data and make predictions
    historic_data = pd.read_csv(historic_data)
    out_df = on_predict(model_and_data, historic_data, args)
    out_df = out_df.sort_values(by=['location', 'time_period'])
    out_df.to_csv(out_file, index=False)
    
    print(f"Predictions saved to: {out_file}")


@app.command()
def plot(model: str, 
         train_data: str, 
         historic_data: str,
         predictions: str,
         model_config: str,
         output: str = "model_visualization.png",
         plot_params: bool = False):
    """Create visualization of model training data, historic data, and predictions."""
    
    config = load_config(model_config)
    
    # Create main visualization
    create_model_visualization(train_data, historic_data, predictions, output, config)
    
    # Optionally create parameter plots
    if plot_params:
        param_output = output.replace('.png', '_parameters.png')
        create_parameter_plot(model, param_output, config)


@app.command()
def plot_components(model: str,
                   model_config: str,
                   output_dir: str = "model_components"):
    """Create specialized plots of model components from posterior samples."""
    
    config = load_config(model_config)
    plot_model_components(model, output_dir, config)


class FileSet(pydantic.BaseModel):
    train_data: str
    historic_data: str
    future_data: str


@pytest.mark.parametrize("folder_name", ['test_data', 'test_data2'])
def test(folder_name):
    fileset = FileSet(
        train_data=('%s/training_data.csv' % folder_name),
        historic_data=('%s/historic_data.csv' % folder_name),
        future_data=('%s/future_data.csv' % folder_name),
    )
    config_filename = 'real_config.yaml'
    train(fileset.train_data,
          'test_runs/model', config_filename)
    predict('test_runs/model',
            fileset.historic_data,
            fileset.future_data,
            'test_runs/forecast_samples.csv',
            config_filename)
    
    # Create visualization
    plot('test_runs/model',
         fileset.train_data,
         fileset.historic_data,
         'test_runs/forecast_samples.csv',
         config_filename,
         f'test_runs/visualization_{folder_name}.png',
         plot_params=True)
    
    df = pd.read_csv('test_runs/forecast_samples.csv')

    for colname in ['location', 'time_period', 'sample_0']:
        assert colname in df.columns
    train_df = pd.read_csv(fileset.train_data)
    future_periods = pd.read_csv(fileset.future_data)['time_period'].unique()
    predicted_periods = df.time_period.unique()
    assert set(future_periods) == set(predicted_periods)
    n_locations = train_df['location'].nunique()
    assert len(df) == n_locations * 3  # 3 horizons


if __name__ == "__main__":
    app()
    # rows = main(args=Args())