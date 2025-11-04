import uuid

import numpy as np
import pandas as pd
import pymc as pm
from pytensor import tensor as pt

from util import (
    complete_monthly_panel,
    extract_month_indices,
    to_tensor_panels,
)


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
