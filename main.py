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
from chap_pymc.models.seasonal_fourier_regression import SeasonalFourierRegression, SeasonalFourierRegressionV2
from chap_pymc.transformations.model_input_creator import FourierInputCreator

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
        content = open(model_config_file).read()
        logger.info(content)
        if model_config_file.endswith('.json'):
            data = json.loads(content)  # type: ignore
        elif model_config_file.endswith('.yaml'):
            data = yaml.load(content, Loader=yaml.FullLoader)
        if 'user_options' not in data:
            data['user_options']  = {}
        data['user_options'] |= {'skip_bottom_n_seasons': 2, 'use_ar': True}
        model_config = ChapConfig.model_validate(data).user_options
    training_df = pd.read_csv(historic_data)
    future_df = pd.read_csv(future_data)
    inference_params = InferenceParams(**model_config.model_dump())
    fourier_hyperparameters = FourierHyperparameters(**model_config.model_dump())
    input_params = FourierInputCreator.Params(**model_config.model_dump())
    assert input_params.skip_bottom_n_seasons == 2, data
    params=SeasonalFourierRegressionV2.Params(inference_params=inference_params,
                                              fourier_hyperparameters=fourier_hyperparameters,
                                              input_params=input_params)
    name = Path(historic_data).stem
    regression_model = SeasonalFourierRegressionV2(params, name=name)
    # model = SeasonalFourierRegression(
    #     prediction_length=3,
    #     lag=3,
    #     fourier_hyperparameters=FourierHyperparameters(**model_config.model_dump()),
    #     inference_params=inference_params
    # )
    predictions = regression_model.predict(training_df, future_df)
    predictions.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")


if __name__ == "__main__":
    app()
