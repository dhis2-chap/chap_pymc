import pydantic

from chap_pymc.curve_parametrizations.fourier_parametrization import FourierHyperparameters
from chap_pymc.inference_params import InferenceParams
from chap_pymc.models.seasonal_regression import SeasonalRegression, ModelParams
from chap_pymc.models.seasonal_fourier_regression import SeasonalFourierRegression
import cyclopts
import pandas as pd
app = cyclopts.App()
#
@app.command()
def train(train_data: str, model: str, model_config: str, force=False):
    return

class FullConfig(InferenceParams, FourierHyperparameters):
    ...

class ChapConfig(pydantic.BaseModel):
    user_options: FullConfig = FullConfig()


@app.command()
def predict(model: str,
            historic_data: str,
            future_data: str,
            out_file: str,
            model_config_file: str | None = None,
):
    """
    Generate predictions using either seasonal or Fourier regression model.

    Args:
        model_type: Type of model to use ('seasonal' or 'fourier')
        historic_data: Path to CSV with historical training data
        future_data: Path to CSV with future data (not used currently)
        out_file: Path to save predictions CSV
        model_config: Optional path to model configuration
        inference_method: Inference method to use ('hmc' or 'advi')
    """
    model_config = FullConfig()
    if model_config_file is not None:
        model_config = ChapConfig.model_validate_json(open(model_config_file).read()).user_options
    training_df = pd.read_csv(historic_data)

    inference_params = InferenceParams(**model_config.model_dump())
    model = SeasonalFourierRegression(
        prediction_length=3,
        lag=3,
        fourier_hyperparameters=FourierHyperparameters(**model_config.model_dump()),
        inference_params=inference_params
    )
    predictions = model.predict(training_df, n_samples=1000)
    predictions.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")


if __name__ == "__main__":
    app()