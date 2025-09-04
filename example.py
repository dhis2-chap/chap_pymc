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
import pytensor

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

def prepare_data(args):
    """Prepare and preprocess data for training."""
    raw = pd.read_csv(args.csv)

    has_rain = args.rain_col in raw.columns
    has_temp = args.temp_col in raw.columns
    keep_cols = [args.target_col]
    if has_rain: keep_cols.append(args.rain_col)
    if has_temp: keep_cols.append(args.temp_col)

    panel, time_idx, locs = complete_monthly_panel(raw, args.date_col, args.loc_col, keep_cols, freq=args.freq)
    T, L, H = len(time_idx), len(locs), args.horizon

    # Target (T, L)
    y = to_tensor_panels(panel, time_idx, locs, args.date_col, args.loc_col, args.target_col)
    if np.isnan(y).any():
        y = safe_impute(y)
    y = np.clip(y, 0, None).astype(int)

    # Covariates (optional) -> standardize
    X_list, feat_names = [], []
    if has_rain:
        X_list.append(to_tensor_panels(panel, time_idx, locs, args.date_col, args.loc_col, args.rain_col))
        feat_names.append(args.rain_col)
    if has_temp:
        X_list.append(to_tensor_panels(panel, time_idx, locs, args.date_col, args.loc_col, args.temp_col))
        feat_names.append(args.temp_col)

    if X_list:
        X = np.stack(X_list, axis=2)  # (T, L, P)
        P = X.shape[2]
        scaler = StandardScaler().fit(X.reshape(T*L, P))
        X_std = scaler.transform(X.reshape(T*L, P)).reshape(T, L, P)
    else:
        P = 0
        X_std = None
        scaler = None

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

        # ----- Independent location effects (no spatial correlation) -----
        sigma_u = pm.HalfNormal("sigma_u", 1.0)

        # ----- AR(1) in time with independent innovations -----
        rho = pm.Uniform("rho", lower=-0.99, upper=0.99)

        # Initial state u0 ~ independent normal
        u0 = pm.Normal("u0", 0.0, sigma_u, shape=L)  # (L,)

        # Independent innovations for T steps (training only)
        v_past = pm.Normal("v_past", 0.0, sigma_u, shape=(T, L))  # (T, L)

        def ar1_step(prev_u, v_t, rho_):
            return rho_ * prev_u + v_t

        u_seq, _ = pytensor.scan(
            fn=ar1_step,
            sequences=[v_past],
            outputs_info=[u0],
            non_sequences=[rho],
        )  # (T, L)

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
    X_std = data_dict['X_std']
    scaler = data_dict['scaler']
    time_idx = data_dict['time_idx']
    locs = data_dict['locs']
    T, L, P, H = data_dict['T'], data_dict['L'], data_dict['P'], data_dict['H']
    
    # Prepare future covariates by holding last value
    if P > 0:
        X_future = np.tile(X_std[-1:, :, :], (H, 1, 1))  # hold last
        Xfut_const = X_future
    else:
        Xfut_const = None

    with model:
        # Extend the model for prediction
        if P:
            beta = model['beta']
            intercept = model['intercept']
            linfut = intercept + pt.tensordot(pt.as_tensor_variable(Xfut_const), beta, axes=[2, 0])
        else:
            intercept = model['intercept']
            linfut = intercept + pt.zeros((H, L))

        # Future innovations
        v_fut = pm.Normal("v_fut", 0.0, model['sigma_u'], shape=(H, L))
        
        # Continue AR(1) from last training state
        # Get the last state from the AR(1) sequence (last time point of training)
        # For simplicity, we'll start from the posterior mean of u0 
        last_u = idata.posterior['u0'].mean(dim=['chain', 'draw']).values
        
        # Generate future sequence
        def ar1_step(prev_u, v_t, rho_):
            return rho_ * prev_u + v_t
            
        u_fut_seq, _ = pytensor.scan(
            fn=ar1_step,
            sequences=[v_fut],
            outputs_info=[pt.as_tensor_variable(last_u)],
            non_sequences=[model['rho']],
        )  # (H, L)

        mu_future = pt.exp(linfut + u_fut_seq)
        y_fut = pm.NegativeBinomial("y_fut", mu=mu_future, alpha=model['alpha'])

        # Sample posterior predictive
        ppc = pm.sample_posterior_predictive(idata, var_names=["y_fut"])

    return ppc

def format_predictions(ppc, data_dict, args):
    """Format predictions into output DataFrame."""
    time_idx = data_dict['time_idx']
    locs = data_dict['locs']
    H = data_dict['H']
    
    # Generate future dates
    fut_dates = pd.date_range(time_idx[-1] + pd.tseries.frequencies.to_offset(args.freq),
                              periods=H, freq=args.freq)
    
    fut_Y = ppc.posterior_predictive["y_fut"]
    # merge chain and draw dims
    fut_Y = fut_Y.stack(draws=("chain", "draw"))  # (H, L, draws)
    fut_dates = [str(p)[:7] for p in fut_dates]
    
    rows = [
        [fut_dates[h], locs[l]] + list(fut_Y[h, l].values.tolist())
        for h in range(fut_Y.shape[0])
        for l in range(fut_Y.shape[1])
    ]
    col_names = ['time_period', 'location'] + [f'Sample_{i+1}' for i in range(fut_Y.shape[2])]
    out_df = pd.DataFrame(rows, columns=col_names)
    out_df.to_csv('forecast_samples.csv', index=False)
    return out_df

def main(args=None):
    """Main function that orchestrates data preparation, training, and prediction."""
    if args is None:
        a = parse_args()
    else:
        a = args
    
    # Prepare data
    data_dict = prepare_data(a)
    
    # Train model
    model, idata = train(data_dict, a)
    
    # Generate predictions
    ppc = predict(model, idata, data_dict, a)
    
    # Format and save results
    out_df = format_predictions(ppc, data_dict, a)
    
    return out_df

class Args(pydantic.BaseModel):
    csv: str = '/home/knut/Data/ch_data/full_data/laos.csv'
    date_col: str = "time_period"
    loc_col: str = "location"
    target_col: str = "disease_cases"
    rain_col: str = "rainfall"
    temp_col: str = "temperature"
    lat_col: str = "lat"
    lon_col: str = "lon"
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


