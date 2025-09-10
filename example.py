# pymc_st_arf_model_nomutable.py
# Temporal PyMC model with Random Walk in time and independent location effects.
# No pm.MutableData / pm.Data usage — just plain NumPy constants.
#
# Usage:
#   python example.py train training_data.csv model.pkl config.yaml
#   python example.py predict model.pkl historic_data.csv future_data.csv predictions.csv config.yaml
#   python example.py plot model.pkl training_data.csv historic_data.csv predictions.csv config.yaml output.png
#   python example.py plot-components model.pkl config.yaml output_dir

import uuid
import pickle

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

from extension import continue_rw_process, prepare_extended_data
from plotting import (
    create_model_visualization,
    create_parameter_plot,
    plot_model_components,
)
from util import  to_tensor_panels, safe_impute, extract_month_indices, complete_monthly_panel

class Config(pydantic.BaseModel):
    covariate_names: list[str] = ['rainfall', 'mean_temperature']
    horizon: int = 3
    tune: int = 200
    draws: int = 200
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
    # Sampling parameters to handle divergences
    target_accept: float = 0.95  # Higher target_accept to reduce divergences
    max_treedepth: int = 12  # Increased max tree depth
    # Non-centered parameterization option
    use_non_centered: bool = False  # Use non-centered parameterization for better sampling
    # Regularization parameters
    beta_prior_sigma: float = 0.5  # Stronger regularization for covariates to reduce correlation with seasonal effects
    random_walk_sigma: float = 0.5

DO_TRAIN = False

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
    if np.isnan(y).any():
        y = safe_impute(y)
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
        if var_name.startswith('y_fut'):
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
        [fut_dates[h], locs[l]] + list(fut_Y[..., h, l, :].values.ravel().tolist())
        for h in range(H)
        for l in range(len(locs))
    ]

    sample_names = [f'sample_{i}' for i in range(args.draws*args.chains)]#fut_Y.shape[-1])]
    assert len(sample_names) == args.draws*args.chains, f"Sample count mismatch: expected {args.draws*args.chains}, got {len(sample_names)}"
    col_names = ['time_period', 'location'] + sample_names
    out_df = pd.DataFrame(rows, columns=col_names)
    
    return out_df

def on_train(training_data: pd.DataFrame, args: Config= Config()) -> tuple:
    '''This should train a model and return everything needed for prediction'''
    """Train the PyMC model and return fitted model with inference data."""
    data_dict = prepare_data(args, training_data)

    y = data_dict["y"]
    X_std = data_dict["X_std"]
    log_pop_offset = data_dict["log_pop_offset"]

    T, L, P, H = data_dict["T"], data_dict["L"], data_dict["P"], data_dict["H"]

    # Convert constants to graph tensors
    y_const = y
    Xpast_const = X_std if P else None
    log_pop_offset_const = log_pop_offset

    month_indices = data_dict["month_indices"]
    # Add new values for prediction period
    new_month_indices = (np.arange(args.horizon)+month_indices[-1]+1)%args.seasonal_periods
    month_indices = np.append(month_indices, new_month_indices)
    log_pop_offset_future = np.tile(log_pop_offset[-1:, :], (H, 1))
    log_pop_offset_const = np.concatenate([log_pop_offset_const, log_pop_offset_future])
    with pm.Model() as model:
        # ----- Fixed effects -----
        linpast = get_linear_predictor(P, Xpast_const, args)
        seasonal_effects = get_seasonal_effects(L, T, args, month_indices)
        location_effects = get_location_effects(L, args)
        u_seq = get_rw_effect(L, T+args.horizon)

        # ----- Observation model: Negative Binomial -----
        alpha = pm.HalfNormal("alpha", 1.0, shape=L)

        # Add population offset to the linear predictor before exponentiating
        without_effect = seasonal_effects + location_effects + u_seq + pt.as_tensor_variable(log_pop_offset_const)
        pm.Deterministic('without_effect', without_effect)
        #decay=pm.Beta("decay", 1, 1)
        for h in range(H+1):
            log_mu_past = pm.Deterministic(
                f"log_mu_past_{h}",
                linpast + without_effect[h:T+h, ...]
            )
            mu_past = pt.exp(log_mu_past)

            y_like = pm.NegativeBinomial(
                f"y_{h}", mu=mu_past[:T-h], alpha=alpha, observed=y_const[h:])

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            max_treedepth=args.max_treedepth,
            random_seed=args.seed,
            progressbar=True,
        )
    # Return everything needed for prediction
    return (model, idata, data_dict, args)


def get_rw_effect(L, T):
    # ----- Random Walk process for each location -----
    sigma_rw = pm.HalfNormal("sigma_rw", 0.1)
    u_seq = pm.GaussianRandomWalk("u_rw",
                                  init_dist=pm.Normal.dist(0, 1),
                                  sigma=sigma_rw,
                                  shape=(T, L))
    return u_seq


def get_location_effects(L, args):
    location_sigma = pm.HalfNormal("location_sigma", args.location_sigma)
    location_effects = pm.Normal("location_raw", 0.0, location_sigma, shape=L)[None, :]
    return location_effects


def get_seasonal_effects(L, T, args, month_indices):
    # ----- Seasonal effects (if enabled) -----
    if not args.use_seasonal_effects:
        seasonal_effects = pt.zeros((T, L))
    if args.use_non_centered:
        # Non-centered parameterization for better sampling
        seasonal_base, seasonal_loc = get_non_centered_seasonal(L, args)
    else:
        # Centered parameterization (original)
        seasonal_base, seasonal_loc = get_centered_seasonal(L, args)

        # Map seasonal effects to time periods using month indices
    total_seasonal = pm.Deterministic('total_seasonal', (seasonal_base+seasonal_loc).T)
    month_tensor = pt.as_tensor_variable(month_indices)
    seasonal_effects = total_seasonal[month_tensor, :]
    # seasonal_base_mapped = seasonal_base[month_tensor]  # (T,)
    # seasonal_loc_mapped = seasonal_loc[month_tensor, :]  # (T, L)
    #
    # # Total seasonal effect = base + location-specific
    # seasonal_effects = seasonal_base_mapped[:, None]  #  + seasonal_loc_mapped
    #total_seasonal = pm.Deterministic("total_seasonal", seasonal_effects)

    return seasonal_effects


def get_centered_seasonal(L, args):
    seasonal_sigma_base = pm.HalfNormal("seasonal_sigma_base", args.seasonal_sigma_base)
    seasonal_base_raw = pm.GaussianRandomWalk(
        "seasonal_base_raw",
        init_dist=pm.Normal.dist(0, 0.1),
        sigma=seasonal_sigma_base,
        shape=args.seasonal_periods,
    )
    seasonal_loc = pm.GaussianRandomWalk(
        name="seasonal_loc",
        init_dist = pm.Normal.dist(np.zeros(L, dtype=float), 0.001),
        sigma=seasonal_sigma_base/2,
        shape = (L, args.seasonal_periods))

    return seasonal_base_raw, seasonal_loc




def get_non_centered_seasonal(L, args):
    seasonal_sigma_base = pm.HalfNormal("seasonal_sigma_base", args.seasonal_sigma_base)
    # Non-centered base seasonal effect
    seasonal_base_raw_std = pm.GaussianRandomWalk(
        "seasonal_base_raw_std",
        init_dist=pm.Normal.dist(0, 0.1),
        sigma=args.random_walk_sigma,
        shape=args.seasonal_periods)

    seasonal_base_raw = seasonal_base_raw_std * seasonal_sigma_base
    seasonal_base = seasonal_base_raw - pt.mean(seasonal_base_raw)

    # Non-centered location-specific seasonal effects
    seasonal_sigma_loc = pm.HalfNormal("seasonal_sigma_loc", args.seasonal_sigma_loc)

    seasonal_loc_raw_std = pm.Normal("seasonal_loc_raw_std", 0.0, 1.0,
                                     shape=(args.seasonal_periods, L))
    seasonal_loc_raw = seasonal_loc_raw_std * seasonal_sigma_loc
    seasonal_loc = seasonal_loc_raw - pt.mean(seasonal_loc_raw, axis=0)
    return seasonal_base, seasonal_loc


def get_linear_predictor(P, Xpast_const, args):
    intercept = pm.Normal("intercept", 0.0, 5.0)
    beta_sigma = args.beta_prior_sigma
    beta = pm.Normal("beta", 0.0, beta_sigma, shape=P)
    linpast = intercept + pt.tensordot(
        pt.as_tensor_variable(Xpast_const), beta, axes=[2, 0]
    )
    return linpast



def new_predict(model_and_data, args: Config):
    model, idata, data_dict, args = model_and_data
    posterior = idata.posterior
    vs = []
    for h in range(1, args.horizon+1):
        vs.append(posterior[f'log_mu_past_{h}'].values[..., [-1], :]) #Last time point
    log_mu_future = np.concatenate(vs, axis=-2)
    alpha = posterior['alpha'].values[..., None, :]
    with pm.Model() as pred_model:
        # Get parameter values from posterior

        mu_future = pt.exp(log_mu_future)
        y_fut = pm.NegativeBinomial(f"y_fut",
                                    mu=mu_future,
                                    alpha=alpha)

        # Sample from the prediction model
        ppc = pm.sample_prior_predictive(draws=1)

    return ppc


def on_predict(model_and_data: tuple, historic_data: pd.DataFrame, args: Config= Config()) -> pd.DataFrame:
    '''Use trained model parameters to predic t from the end of historic_data'''
    # Unpack the training results
    #if DO_TRAIN:
    model, idata, data_dict, args = model_and_data
    extended_data_dict = prepare_extended_data(data_dict, historic_data, args)
    ppc = new_predict(model_and_data, args)
    result_df = format_predictions_from_historic_end(ppc, extended_data_dict, args)
    return result_df


    #else:

    # Prepare extended data that includes historic observations

    
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
            # Get seasonal parameters from posterior - handle both centered and non-centered
            total_seasonal = idata.posterior['total_seasonal'].mean(dim=["chain", "draw"]).values

            # Generate future month indices starting from end of historic data
            extended_time_idx = extended_data_dict['extended_time_idx']
            last_date = pd.Timestamp(extended_time_idx[-1])
            future_start_date = last_date + pd.tseries.frequencies.to_offset(args.freq)
            future_dates = pd.date_range(future_start_date, periods=H, freq=args.freq)
            future_month_indices = extract_month_indices(future_dates)
            
            # Map seasonal effects to future periods
            seasonal_effects_fut = total_seasonal[future_month_indices, :]  # (H,)

        # Location effects for future periods
        # Get location parameters from posterior
        location_effects_fut = idata.posterior["location_raw"].mean(dim=["chain", "draw"]).values[None, :]
        #location_val = location_val - np.mean(location_val)  # Apply zero-sum constraint
            
        # Location effects are constant across time, so broadcast to (H, L)
        # location_effects_fut = pt.as_tensor_variable(np.tile(location_val[None, :], (H, 1)))

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

        # Add population offset to future predictionsw
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




app = cyclopts.App()

@app.command()
def train(train_data: str, model: str, model_config: str, force=False):

    if (not DO_TRAIN) and not force:
        return
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
    print(f"Predicting {model}")
    if not DO_TRAIN:
        print('Training first')
        train(historic_data, model, model_config, force=True)

    base_name = model.rsplit(".", 1)[0] if "." in model else model
    idata_filename = f"{base_name}_idata.nc"
    args = load_config(model_config)
    # Load inference data from NetCDF
    idata = az.from_netcdf(idata_filename)

    # Load data dictionary and config from pickle
    data_filename = f"{base_name}_data.pkl"
    with open(data_filename, 'rb') as f:
        data_dict, args = pickle.load(f)

    # Create a dummy model object (we don't need the actual model for prediction)
    model = None
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
    config_filename = 'test_config.yaml'
    train(fileset.train_data,
          'test_runs/model', config_filename)
    predict(
        "test_runs/model",
        fileset.historic_data,
        fileset.future_data,
        "test_runs/forecast_samples.csv",
        config_filename,
    )

    plot_components('test_runs/model', config_filename)


    # Create visualization
    # plot('test_runs/model',
    #      fileset.train_data,
    #      fileset.historic_data,
    #      'test_runs/forecast_samples.csv',
    #      config_filename,
    #      f'test_runs/visualization_{folder_name}.png',
    #      plot_params=True)
    #
    # df = pd.read_csv('test_runs/forecast_samples.csv')
    #
    # for colname in ['location', 'time_period', 'sample_0']:
    #     assert colname in df.columns
    # train_df = pd.read_csv(fileset.train_data)
    # future_periods = pd.read_csv(fileset.future_data)['time_period'].unique()
    # predicted_periods = df.time_period.unique()
    # assert set(future_periods) == set(predicted_periods)
    # n_locations = train_df['location'].nunique()
    # assert len(df) == n_locations * 3  # 3 horizons


if __name__ == "__main__":
    app()
    # rows = main(args=Args())