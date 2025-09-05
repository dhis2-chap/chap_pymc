# pymc_st_arf_model_nomutable.py
# Temporal PyMC model with AR(1) in time and independent location effects.
# No pm.MutableData / pm.Data usage — just plain NumPy constants.
#
# Usage:
#   python pymc_st_arf_model_nomutable.py --csv your_file.csv

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

def continue_ar_process(idata, T_orig, T_extended, sigma_u_val, rho_val, L):
    """Continue AR(1) process from training end through historic period."""
    if T_extended <= T_orig:
        # No extension needed
        u_ar_last = (
            idata.posterior["u_ar"]
            .isel(u_ar_dim_0=-1)
            .mean(dim=["chain", "draw"])
            .values
        )
        return pt.as_tensor_variable(u_ar_last)
    
    # Get starting point from training
    u_ar_last = (
        idata.posterior["u_ar"]
        .isel(u_ar_dim_0=-1)
        .mean(dim=["chain", "draw"])
        .values
    )
    
    # Continue AR process for the extended period
    steps_to_continue = T_extended - T_orig
    u_current = pt.as_tensor_variable(u_ar_last)
    
    # Generate AR steps for the historic extension period
    # Note: We're using the mean behavior here, not sampling noise
    for step in range(steps_to_continue):
        # For deterministic continuation, we could omit noise, but let's include it
        # to maintain stochastic behavior
        unique_id = str(uuid.uuid4())[:8]
        noise = pm.Normal(f"historic_noise_{unique_id}_{step}", 0.0, sigma_u_val, shape=(L,))
        u_current = rho_val * u_current + noise
    
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
            seasonal_sigma_base = pm.HalfNormal("seasonal_sigma_base", args.seasonal_sigma_base)
            
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

        # ----- AR(1) process using PyMC's built-in AR distribution -----
        rho = pm.Uniform("rho", lower=-0.99, upper=0.99)
        sigma_u = pm.HalfNormal("sigma_u", 1.0)

        # AR(1) process for each location independently
        # pm.AR expects shape (T, L) and creates an AR process along the time dimension
        u_seq = pm.AR("u_ar", rho=rho, sigma=sigma_u, shape=(T, L))

        # ----- Observation model: Negative Binomial -----
        alpha = pm.HalfNormal("alpha", 1.0)
        # Add population offset to the linear predictor before exponentiating
        log_mu_past = linpast + seasonal_effects + u_seq + pt.as_tensor_variable(log_pop_offset_const)
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
    '''Use trained model parameters to predict from the end of historic_data'''
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
        sigma_u_val = idata.posterior["sigma_u"].mean(dim=["chain", "draw"]).values
        rho_val = idata.posterior["rho"].mean(dim=["chain", "draw"]).values
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

        # Continue AR(1) process from training through historic period to get final state
        u_ar_extended = continue_ar_process(idata, T_orig, T_extended, sigma_u_val, rho_val, L)

        # Create future AR process starting from end of historic period
        unique_id = str(uuid.uuid4())[:8]

        # For prediction, we'll use a simple AR(1) continuation
        # Initialize with last state after processing historic data
        u_fut = [u_ar_extended]

        # Generate H future steps
        for h in range(H):
            noise = pm.Normal(f"noise_{unique_id}_{h}", 0.0, sigma_u_val, shape=(L,))
            next_u = rho_val * u_fut[-1] + noise
            u_fut.append(next_u)

        # Stack future AR values (skip initial value)
        u_fut_seq = pt.stack(u_fut[1:], axis=0)  # (H, L)

        # Add population offset to future predictions
        log_mu_future = linfut + seasonal_effects_fut + u_fut_seq + pt.as_tensor_variable(log_pop_offset_future)
        mu_future = pt.exp(log_mu_future)
        y_fut = pm.NegativeBinomial(f"y_fut_{unique_id}", mu=mu_future, alpha=alpha_val)

        # Sample from the prediction model
        ppc = pm.sample_prior_predictive(
            samples=len(idata.posterior.chain) * len(idata.posterior.draw)
        )

    # Format predictions starting from end of historic data
    result_df = format_predictions_from_historic_end(ppc, extended_data_dict, args)
    
    return result_df


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