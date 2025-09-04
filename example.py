# pymc_st_arf_model_nomutable.py
# Temporal PyMC model with AR(1) in time and independent location effects.
# No pm.MutableData / pm.Data usage — just plain NumPy constants.
#
# Usage:
#   python pymc_st_arf_model_nomutable.py --csv your_file.csv

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pydantic
import pymc as pm
import pytensor.tensor as pt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", default="time_period")
    ap.add_argument("--loc-col", default="location")
    ap.add_argument("--target-col", default="disease_cases")
    ap.add_argument("--rain-col", default="rainfall")
    ap.add_argument("--temp-col", default="temperature")
    ap.add_argument("--lat-col", default="lat")
    ap.add_argument("--lon-col", default="lon")
    ap.add_argument("--freq", default="MS")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

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

def prepare_data(args, raw):
    """Prepare and preprocess data for training."""

    keep_cols = [args.target_col, args.rain_col, args.temp_col]
    panel, time_idx, locs = complete_monthly_panel(raw, args.date_col, args.loc_col, keep_cols, freq=args.freq)
    T, L, H = len(time_idx), len(locs), args.horizon

    # Target (T, L)
    y = to_tensor_panels(panel, time_idx, locs, args.date_col, args.loc_col, args.target_col)
    if np.isnan(y).any():
        y = safe_impute(y)
    y = np.clip(y, 0, None).astype(int)

    # Covariates (optional) -> standardize
    feat_names = [args.rain_col, args.temp_col]
    X_list = [to_tensor_panels(panel, time_idx, locs, args.date_col, args.loc_col, col) for col in [args.rain_col, args.temp_col]]
    # X_list.append(to_tensor_panels(panel, time_idx, locs, args.date_col, args.loc_col, args.rain_col))
    # feat_names.append(args.rain_col)
    # X_list.append(to_tensor_panels(panel, time_idx, locs, args.date_col, args.loc_col, args.temp_col))
    # feat_names.append(args.temp_col)
    X = np.stack(X_list, axis=2)  # (T, L, P)
    P = X.shape[2]
    scaler = StandardScaler().fit(X.reshape(T*L, P))
    X_std = scaler.transform(X.reshape(T*L, P)).reshape(T, L, P)
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
        'H': H
    }


def train(data_dict, args):
    """Train the PyMC model and return fitted model with inference data."""
    y = data_dict['y']
    X_std = data_dict['X_std']
    T, L, P, H = data_dict['T'], data_dict['L'], data_dict['P'], data_dict['H']
    
    # Convert constants to graph tensors
    y_const = y
    Xpast_const = X_std if P else None

    with pm.Model() as model:
        # ----- Fixed effects -----
        intercept = pm.Normal("intercept", 0.0, 5.0)
        if P:
            beta = pm.Normal("beta", 0.0, 1.0, shape=P)
            linpast = intercept + pt.tensordot(pt.as_tensor_variable(Xpast_const), beta, axes=[2, 0])
        else:
            linpast = intercept + pt.zeros((T, L))

        # ----- AR(1) process using PyMC's built-in AR distribution -----
        rho = pm.Uniform("rho", lower=-0.99, upper=0.99)
        sigma_u = pm.HalfNormal("sigma_u", 1.0)
        
        # AR(1) process for each location independently
        # pm.AR expects shape (T, L) and creates an AR process along the time dimension
        u_seq = pm.AR("u_ar", rho=rho, sigma=sigma_u, shape=(T, L))

        # ----- Observation model: Negative Binomial -----
        alpha = pm.HalfNormal("alpha", 1.0)
        mu_past = pt.exp(linpast + u_seq)
        y_like = pm.NegativeBinomial("y", mu=mu_past, alpha=alpha, observed=y_const)

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=0.9,
            random_seed=args.seed,
            progressbar=True,
        )

    return model, idata

def predict(model, idata, data_dict, args):
    """Generate predictions using the trained model."""
    import uuid
    X_std = data_dict['X_std']
    T, L, P, H = data_dict['T'], data_dict['L'], data_dict['P'], data_dict['H']
    
    # Prepare future covariates by holding last value
    if P > 0:
        X_future = np.tile(X_std[-1:, :, :], (H, 1, 1))  # hold last
        Xfut_const = X_future
    else:
        Xfut_const = None

    # Create a new model for prediction to avoid variable name conflicts
    with pm.Model() as pred_model:
        # Get parameter values from posterior
        intercept_val = idata.posterior['intercept'].mean(dim=['chain', 'draw']).values
        sigma_u_val = idata.posterior['sigma_u'].mean(dim=['chain', 'draw']).values
        rho_val = idata.posterior['rho'].mean(dim=['chain', 'draw']).values
        alpha_val = idata.posterior['alpha'].mean(dim=['chain', 'draw']).values
        
        # Fixed effects for future periods
        if P:
            beta_val = idata.posterior['beta'].mean(dim=['chain', 'draw']).values
            linfut = intercept_val + pt.tensordot(pt.as_tensor_variable(Xfut_const), pt.as_tensor_variable(beta_val), axes=[2, 0])
        else:
            linfut = intercept_val + pt.zeros((H, L))
        
        # Get the last values from the trained AR process
        # Extract last time step from the AR process
        u_ar_last = idata.posterior['u_ar'].isel(u_ar_dim_0=-1).mean(dim=['chain', 'draw']).values  # (L,)
        
        # Create future AR process starting from last observed state
        unique_id = str(uuid.uuid4())[:8]
        
        # For prediction, we'll use a simple AR(1) continuation
        # Initialize with last observed values
        u_fut = [pt.as_tensor_variable(u_ar_last)]
        
        # Generate H future steps
        for h in range(H):
            noise = pm.Normal(f"noise_{unique_id}_{h}", 0.0, sigma_u_val, shape=(L,))
            next_u = rho_val * u_fut[-1] + noise
            u_fut.append(next_u)
        
        # Stack future AR values (skip initial value)
        u_fut_seq = pt.stack(u_fut[1:], axis=0)  # (H, L)

        mu_future = pt.exp(linfut + u_fut_seq)
        y_fut = pm.NegativeBinomial(f"y_fut_{unique_id}", mu=mu_future, alpha=alpha_val)

        # Sample from the prediction model
        ppc = pm.sample_prior_predictive(samples=len(idata.posterior.chain) * len(idata.posterior.draw))

    return ppc

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
    col_names = ['time_period', 'location'] + [f'Sample_{i+1}' for i in range(fut_Y.shape[-1])]
    out_df = pd.DataFrame(rows, columns=col_names)
    out_df.to_csv('forecast_samples.csv', index=False)
    return out_df

def on_train(training_data: pd.DataFrame) -> tuple:
    '''This should train a model and return everything needed for prediction'''
    # We need to extract args from somewhere - for now, use default Args
    # In a real implementation, you might want to pass args differently
    args = Args(tune=1, draws=1, chains=2)
    
    # Prepare data directly
    # Prepare data and train model
    data_dict = prepare_data(args, training_data)
    model, idata = train(data_dict, args)

    # Return everything needed for prediction
    return (model, idata, data_dict, args)

def on_predict(model_and_data: tuple, historic_data: pd.DataFrame) -> pd.DataFrame:
    '''This shoud use the model and data generated during training to predict future disease counts based on the historic data'''
    # Unpack the training results
    model, idata, data_dict, args = model_and_data
    
    # For this implementation, we'll use the original data_dict for prediction
    # The historic_data parameter could be used to update covariates if needed
    # But for now, we'll proceed with the existing approach
    
    # Generate predictions using the trained model
    ppc = predict(model, idata, data_dict, args)
    
    # Format predictions into DataFrame
    result_df = format_predictions(ppc, data_dict, args)
    
    return result_df


def main(args=None):
    """Main function that orchestrates training and prediction using on_train and on_predict."""
    if args is None:
        a = parse_args()
    else:
        a = args
    
    # Load training data
    training_data = pd.read_csv(a.csv)
    
    # Train model using on_train
    model_and_data = on_train(training_data)
    
    # Generate predictions using on_predict 
    # For now, we'll pass the same training data as historic_data
    # In practice, this could be different/updated data
    out_df = on_predict(model_and_data, training_data)
    
    return out_df

class Args(pydantic.BaseModel):
    csv: str = '/home/knut/Data/ch_data/full_data/laos.csv'
    date_col: str = "time_period"
    loc_col: str = "location"
    target_col: str = "disease_cases"
    rain_col: str = "rainfall"
    temp_col: str = "mean_temperature"
    freq: str = "MS"
    horizon: int = 3
    tune: int = 10
    draws: int = 10
    chains: int = 2
    seed: int = 42

def test():
    df = main(args=Args(tune=1, draws=1, chains=2))
    for colname in ['location', 'time_period', 'Sample_1']:
        assert colname in df.columns

if __name__ == "__main__":
    rows = main(args=Args())


