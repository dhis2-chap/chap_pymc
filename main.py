from chap_pymc.mcmc_params import MCMCParams
from chap_pymc.models.seasonal_regression import SeasonalRegression, ModelParams
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
            model_config: str | None = None):
    training_df = pd.read_csv(historic_data)
    model = SeasonalRegression(mcmc_params=MCMCParams(n_iterations=200_000),)
    #model_params=ModelParams(use_mixture=True, mask_empty_seasons=True))

    predictions = model.predict_with_dims(training_df)
    #predictions= model.predict_advi(training_df)
    # predictions = get_predictions(training_df)
    predictions.to_csv(out_file, index=False)


if __name__ == "__main__":
    app()