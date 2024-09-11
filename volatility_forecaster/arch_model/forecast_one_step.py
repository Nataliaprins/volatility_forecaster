import os

import mlflow.pyfunc

from volatility_forecaster.arch_model.extract_serie import extract_serie

# TODO: definir función que traerá los datos


def predict_one_step(stock_name, logged_model_path, column_name):
    data = extract_serie(stock_name, column_name=column_name)
    loaded_model = mlflow.pyfunc.load_model(logged_model_path)
    prediction = loaded_model.predict(data)
    print(prediction)
    return prediction


if __name__ == "__main__":
    predict_one_step(
        stock_name="googl",
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/449779083138429514/5c7d7c596dd54552911df045b2a55300/artifacts/artifacts",
        column_name="log_yield",
    )
