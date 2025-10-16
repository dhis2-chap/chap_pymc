from chap_pymc.mcmc_params import MCMCParams
from chap_pymc.models.seasonal_regression import SeasonalRegression, ModelParams
from chap_pymc.models.seasonal_fourier_regression import SeasonalFourierRegression
import cyclopts
import pandas as pd
app = cyclopts.App()
#
@app.command()
def train(train_data: str, model: str, model_config: str, force=False):
    return

@app.command()
def predict(model: str,
            historic_data: str,
            future_data: str,
            out_file: str,
            model_config: str | None = None,
            model_type: str = 'fourier'
):
    """
    Generate predictions using either seasonal or Fourier regression model.

    Args:
        model_type: Type of model to use ('seasonal' or 'fourier')
        historic_data: Path to CSV with historical training data
        future_data: Path to CSV with future data (not used currently)
        out_file: Path to save predictions CSV
        model_config: Optional path to model configuration
    """
    training_df = pd.read_csv(historic_data)

    if model_type == 'fourier':
        # Fourier-based seasonal model
        model = SeasonalFourierRegression(
            prediction_length=3,
            lag=3,
            n_harmonics=3,
            mcmc_params=MCMCParams(chains=4, tune=500, draws=500)
        )
        predictions = model.predict(training_df, n_samples=1000)
    else:
        # Traditional seasonal regression model
        model = SeasonalRegression(
            mcmc_params=MCMCParams(n_iterations=200_000),
        )
        predictions = model.predict_with_dims(training_df)

    predictions.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")


if __name__ == "__main__":
    app()