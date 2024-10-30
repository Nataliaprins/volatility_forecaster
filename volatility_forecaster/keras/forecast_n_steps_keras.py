import os

import mlflow.tensorflow
import pandas as pd

from volatility_forecaster.constants import ROOT_DIR_PROJECT


def forecast_n_step_keras(
    project_name,
    stock_name,
    logged_model_path,
    n_steps,
):
    # TODO:cambiar ruta a carpeta con datos nuevos
    path = os.path.join(
        ROOT_DIR_PROJECT, "data", project_name, "processed/train_test/keras"
    )

    x_features = pd.read_csv(os.path.join(path, f"{stock_name}_xtest.csv"))
    loaded_model = mlflow.tensorflow.load_model(logged_model_path)
    prediction = loaded_model.predict(x_features[-n_steps:])
    print(prediction)
    return prediction


if __name__ == "__main__":
    forecast_n_step_keras(
        project_name="yahoo",
        stock_name="googl",
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/720987901295834756/b517f88e6930442c9ba263413b670fcc/artifacts/model",
        n_steps=5,
    )
