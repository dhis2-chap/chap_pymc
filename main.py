import json
import logging
from pathlib import Path

import cyclopts
import pandas as pd
import yaml

from chap_pymc.configs.chap_config import ChapConfig, FullConfig
from chap_pymc.curve_parametrizations.fourier_parametrization import (
    FourierHyperparameters,
)
from chap_pymc.inference_params import InferenceParams
from chap_pymc.models.seasonal_fourier_regression import (
    SeasonalFourierRegressionV2,
)
from chap_pymc.transformations.model_input_creator import FourierInputCreator


def detect_frequency(df: pd.DataFrame) -> str:
    """Detect data frequency from time_period format.

    Returns 'W' for weekly data, 'M' for monthly data.
    """
    sample_period = str(df['time_period'].iloc[0])

    # Date range format indicates weekly data
    if '/' in sample_period:
        return 'W'

    # ISO week format (e.g., "2024-W15")
    if 'w' in sample_period.lower():
        return 'W'

    # Check if period number > 12 (must be weekly)
    parts = sample_period.split('-')
    if len(parts) == 2:
        period = int(parts[1])
        if period > 12:
            return 'W'

    return 'M'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
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
            save_plot: bool = False,
            save_data: bool = False,
            ):
    """
    Generate predictions using either seasonal or Fourier regression model.

    Args:
        model_type: Type of model to use ('seasonal' or 'fourier')
        historic_data: Path to CSV with historical training data
        future_data: Path to CSV with future data (not used currently)
        out_file: Path to save predictions CSV
        parsed_model_config: Optional path to model configuration
        inference_method: Inference method to use ('hmc' or 'advi')
        save_plot: Whether to save diagnostic plots (default: False)
        save_data: Whether to save intermediate data files (default: False)
    """
    parsed_model_config = FullConfig()
    if model_config is not None:
        content = open(model_config).read()
        logger.info(content)
        if model_config.endswith('.json'):
            data = json.loads(content)  # type: ignore
        elif model_config.endswith('.yaml'):
            data = yaml.load(content, Loader=yaml.FullLoader)
        if 'user_options' not in data:
            data['user_options']  = {}
        parsed_model_config = ChapConfig.model_validate(data).user_options
    training_df = pd.read_csv(historic_data)
    future_df = pd.read_csv(future_data)

    # Detect data frequency and set in params
    frequency = detect_frequency(training_df)
    logger.info(f"Detected data frequency: {frequency}")

    inference_params = InferenceParams(**parsed_model_config.model_dump())
    fourier_hyperparameters = FourierHyperparameters(**parsed_model_config.model_dump())

    # Create input params with detected frequency
    #seasonal_params = SeasonalXArray.Params(frequency=frequency)
    input_params = FourierInputCreator.Params(
        **parsed_model_config.model_dump(),
        #seasonal_params=seasonal_params
    )
    input_params.seasonal_params.frequency = frequency
    params=SeasonalFourierRegressionV2.Params(inference_params=inference_params,
                                              fourier_hyperparameters=fourier_hyperparameters,
                                              input_params=input_params)
    name = Path(historic_data).stem if save_data else None
    regression_model = SeasonalFourierRegressionV2(params, name=name)
    # Note: save_plot will be skipped if name contains invalid path characters
    # model = SeasonalFourierRegression(
    #     prediction_length=3,
    #     lag=3,
    #     fourier_hyperparameters=FourierHyperparameters(**parsed_model_config.model_dump()),
    #     inference_params=inference_params
    # )
    predictions = regression_model.predict(training_df, future_df, save_plot=save_plot)
    predictions.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")


if __name__ == "__main__":
    app()
