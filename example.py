# pymc_st_arf_model_nomutable.py
# Spatio-temporal PyMC model with AR(1) in time and spatial GP innovations.
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

def main(args=None):
    if args is None:
        a = parse_args()
    else:
        a = args
    raw = pd.read_csv(a.csv)

    has_rain = a.rain_col in raw.columns
    has_temp = a.temp_col in raw.columns
    has_latlon = (a.lat_col in raw.columns) and (a.lon_col in raw.columns)

    keep_cols = [a.target_col]
    if has_rain: keep_cols.append(a.rain_col)
    if has_temp: keep_cols.append(a.temp_col)
    if has_latlon:
        loc_latlon = raw[[a.loc_col, a.lat_col, a.lon_col]].drop_duplicates(a.loc_col)

    panel, time_idx, locs = complete_monthly_panel(raw, a.date_col, a.loc_col, keep_cols, freq=a.freq)
    T, L, H = len(time_idx), len(locs), a.horizon

    # Target (T, L)
    y = to_tensor_panels(panel, time_idx, locs, a.date_col, a.loc_col, a.target_col)
    if np.isnan(y).any():
        y = safe_impute(y)
    y = np.clip(y, 0, None).astype(int)

    # Covariates (optional) -> standardize -> hold last value for future H months
    X_list, feat_names = [], []
    if has_rain:
        X_list.append(to_tensor_panels(panel, time_idx, locs, a.date_col, a.loc_col, a.rain_col))
        feat_names.append(a.rain_col)
    if has_temp:
        X_list.append(to_tensor_panels(panel, time_idx, locs, a.date_col, a.loc_col, a.temp_col))
        feat_names.append(a.temp_col)

    if X_list:
        X = np.stack(X_list, axis=2)  # (T, L, P)
        P = X.shape[2]
        scaler = StandardScaler().fit(X.reshape(T*L, P))
        X_std = scaler.transform(X.reshape(T*L, P)).reshape(T, L, P)
        X_future = np.tile(X_std[-1:, :, :], (H, 1, 1))  # hold last
    else:
        P = 0
        X_std = None
        X_future = None

    # Spatial coordinates (L, 2) if available
    latlon = None
    if has_latlon:
        latlon = np.zeros((L, 2))
        ll = loc_latlon.set_index(a.loc_col).loc[locs]
        latlon[:, 0] = ll[a.lat_col].to_numpy()
        latlon[:, 1] = ll[a.lon_col].to_numpy()

    # Convert constants to graph tensors
    y_const = y
    Xpast_const = X_std if P else None
    Xfut_const  = X_future if P else None

    with pm.Model() as m:
        # ----- Fixed effects -----
        intercept = pm.Normal("intercept", 0.0, 5.0)
        if P:
            beta = pm.Normal("beta", 0.0, 1.0, shape=P)
            linpast = intercept + pt.tensordot(pt.as_tensor_variable(Xpast_const), beta, axes=[2, 0])
            linfut  = intercept + pt.tensordot(pt.as_tensor_variable(Xfut_const),  beta, axes=[2, 0])
        else:
            linpast = intercept + pt.zeros((T, L))
            linfut  = intercept + pt.zeros((H, L))

        # ----- Spatial kernel (GP over locations) or iid fallback -----
        if latlon is not None:
            ell = pm.HalfNormal("ell", 2.0)
            amp = pm.HalfNormal("amp", 1.0)
            cov = pm.gp.cov.Matern52(2, ls=ell)     # ls arg name is widely supported
            K = (amp**2) * cov(latlon) + 1e-6 * pt.eye(L)
        else:
            amp = pm.HalfNormal("amp", 1.0)
            K = (amp**2) * pt.eye(L)

        K_chol = pt.linalg.cholesky(K)

        # ----- AR(1) in time with spatially correlated innovations -----
        rho = pm.Uniform("rho", lower=-0.99, upper=0.99)

        # Initial state u0 ~ MVN(0, K)
        z0 = pm.Normal("z0", 0.0, 1.0, shape=L)
        u0 = pt.dot(K_chol, z0)  # (L,)

        # Innovations for all T+H steps
        z_full = pm.Normal("z_full", 0.0, 1.0, shape=(T + H, L))
        v_full = pt.dot(z_full, K_chol.T)  # (T+H, L)

        def ar1_step(prev_u, v_t, rho_):
            return rho_ * prev_u + v_t

        u_seq, _ = pytensor.scan(
            fn=ar1_step,
            sequences=[v_full],
            outputs_info=[u0],
            non_sequences=[rho],
        )  # (T+H, L)

        u_past = u_seq[:T, :]
        u_fut  = u_seq[T:, :]

        # ----- Observation model: Negative Binomial -----
        alpha = pm.HalfNormal("alpha", 1.0)
        mu_past = pt.exp(linpast + u_past)
        y_like = pm.NegativeBinomial("y", mu=mu_past, alpha=alpha, observed=y_const)

        mu_future = pt.exp(linfut + u_fut)
        y_fut = pm.NegativeBinomial("y_fut", mu=mu_future, alpha=alpha)

        idata = pm.sample(
            draws=a.draws,
            tune=a.tune,
            chains=a.chains,
            target_accept=0.9,
            random_seed=a.seed,
            progressbar=True,
        )

        ppc = pm.sample_posterior_predictive(idata, var_names=["y_fut"])

    # ----- Summarize forecasts -----
    fut_dates = pd.date_range(time_idx[-1] + pd.tseries.frequencies.to_offset(a.freq),
                              periods=H, freq=a.freq)
    fut_Y = ppc.posterior_predictive["y_fut"]
    print(fut_Y)
    print(type(fut_Y))
    # merge chain and draw dims
    fut_Y = fut_Y.stack(draws=("chain", "draw"))  # (S, H, L)
    fut_dates = [str(p)[:7] for p in fut_dates]
    rows = [
        [fut_dates[p], locs[l]] + list(fut_Y[p, l].values.tolist())
        for p in range(fut_Y.shape[0])
        for l in range(fut_Y.shape[1])
    ]
    col_names = ['time_period', 'location'] + [f'Sample_{i+1}' for i in range(fut_Y.shape[2])]
    out_df = pd.DataFrame(rows, columns=col_names)
    out_df.to_csv('forecast_samples.csv', index=False)
    return out_df
    post_y = fut_Y.stack(draw=("chain", "draw")).values  # (S, H, L)
    mean_pred = post_y.mean(axis=0)  # (H, L)

    rows = []
    for h in range(H):
        for j, loc in enumerate(locs):
            rows.append({
                "location": loc,
                "horizon": h + 1,
                "forecast_date": fut_dates[h].strftime("%Y-%m-%d"),
                "pred_cases_mean": float(mean_pred[h, j]),
            })
    out = pd.DataFrame(rows).sort_values(["location", "horizon"])
    print(out.head(12).to_string(index=False))

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


