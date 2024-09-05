"this function forecasts n steps ahead using a registered model in mlflow"


import mlflow.pyfunc
import numpy as np

from volatility_forecaster.arch_model.extract_serie import extract_serie


def forecast_n_steps(
    stock_name,
    logged_model_path,
    column_name,
    n_steps,
):
    data = extract_serie(stock_name, column_name=column_name)
    data_numpy = data.to_numpy().reshape(-1, 1)

    for _ in range(n_steps):
        # load the model
        loaded_model = mlflow.pyfunc.load_model(logged_model_path)
        # predict one step
        prediction = loaded_model.predict(data_numpy)
        prediction_numpy = prediction.to_numpy().reshape(-1, 1)
        # append the prediction to the data
        prediction_numpy = np.append(data_numpy, prediction_numpy, axis=0)
        print(prediction_numpy)
        # update the data
        data_numpy = prediction_numpy
        print(data_numpy)

    return prediction_numpy


if __name__ == "__main__":
    forecast_n_steps(
        stock_name="googl",
        logged_model_path="/Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/546022347724216931/1e69094578184605b5bf4804fe873c96/artifacts/artifacts",
        column_name="log_yield",
        n_steps=5,
    )
