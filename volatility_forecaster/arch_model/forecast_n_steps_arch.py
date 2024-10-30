"this function forecasts n steps ahead using a registered model in mlflow"


import mlflow.pyfunc
import numpy as np

from volatility_forecaster.preprocessing.extract_serie import extract_serie


def forecast_n_steps_arch(
    project_name,
    stock_name,
    logged_model_path,
    column_name,
    n_steps,
):
    data = extract_serie(stock_name, project_name=project_name, column_name=column_name)
    data_numpy = data.to_numpy().reshape(-1, 1)

    for _ in range(n_steps):
        # load the model
        loaded_model = mlflow.pyfunc.load_model(logged_model_path)
        # predict one step
        prediction = loaded_model.predict(data_numpy)
        prediction_numpy = prediction.to_numpy().reshape(-1, 1)
        # append the prediction to the data
        prediction_numpy = np.append(data_numpy, prediction_numpy, axis=0)
        # update the data
        data_numpy = prediction_numpy
        print(data_numpy[-n_steps:])

    return prediction_numpy


if __name__ == "__main__":
    forecast_n_steps_arch(
        project_name="yahoo",
        stock_name="googl",
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/158973272966166127/197bedaf8fde4057bb4ea53b75e30f5f/artifacts/model",
        column_name="log_yield",
        n_steps=5,
    )
