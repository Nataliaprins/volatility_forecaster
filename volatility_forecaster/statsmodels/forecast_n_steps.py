"this function forecasts n steps ahead using a registered model in mlflow"

import mlflow.statsmodels
import numpy as np
import pandas as pd

from volatility_forecaster.preprocessing.extract_serie import extract_serie


def forecast_n_steps(
    project_name,
    stock_name,
    logged_model_path,
    column_name,
    lags,
    n_steps,
):
    data = extract_serie(stock_name, project_name=project_name, column_name=column_name)
    data_numpy = data.to_numpy().reshape(-1, 1)

    for _ in range(lags):
        # load the model
        loaded_model = mlflow.statsmodels.load_model(logged_model_path)
        forecast = loaded_model.forecast(steps=n_steps)
        print(forecast)

        return

    return


if __name__ == "__main__":
    forecast_n_steps(
        project_name="yahoo",
        stock_name="googl",
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/158973272966166127/e163c8aadf5645c59e9847f23cbbe20f/artifacts/model",
        column_name="log_yield",
        lags=7,
        n_steps=7,
    )
