"""Tests for the temporal PyMC model (example.py)."""
import pytest
import pydantic

from examples.seasonal_regression_cli import train, predict, plot_components


class FileSet(pydantic.BaseModel):
    train_data: str
    historic_data: str
    future_data: str


@pytest.mark.skip(reason="Requires test_data and test_data2 directories with historic_data.csv and future_data.csv which are not in the repository")
@pytest.mark.parametrize("folder_name", ['test_data', 'test_data2'])
def test_temporal_model_workflow(folder_name):
    """Test the full workflow: train, predict, and plot components."""
    fileset = FileSet(
        train_data=(f'{folder_name}/training_data.csv'),
        historic_data=(f'{folder_name}/historic_data.csv'),
        future_data=(f'{folder_name}/future_data.csv'),
    )
    config_filename = 'tests/fixtures/config/test_config.yaml'

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
