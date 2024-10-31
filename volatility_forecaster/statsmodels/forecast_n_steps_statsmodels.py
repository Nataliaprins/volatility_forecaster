"this function forecasts n steps ahead using a registered model in mlflow"

import os

import mlflow.statsmodels
import pandas as pd

from volatility_forecaster.preprocessing.extract_serie import extract_serie


def forecast_n_steps_statsmodels(
    project_name,
    stock_name,
    logged_model_path,
    n_steps,
):
    test_data = pd.read_csv(
        os.path.join(
            "/Users/nataliaacevedo/volatility_forecaster/",
            "data",
            project_name,
            "processed",
            "train_test",
            "statsmodels",
            f"statsmodels_{stock_name}_test.csv",
        )
    )

    # load the model
    loaded_model = mlflow.statsmodels.load_model(logged_model_path)
    forecast = loaded_model.forecast(steps=(len(test_data) + n_steps))
    return forecast.tail(n_steps)


if __name__ == "__main__":
    forecast_n_steps_statsmodels(
        project_name="yahoo",
        stock_name="googl",
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/366621011459063161/8cdf99170e8a47b1993864b931647c32/artifacts/model",
        n_steps=7,
    )
