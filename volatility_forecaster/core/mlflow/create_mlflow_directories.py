import os

from volatility_forecaster.constants import ROOT_DIR_PROJECT


def create_mlflow_directories(project_name: str):
    if not os.path.exists(
        os.path.join(ROOT_DIR_PROJECT, "data", project_name, "models", "mlflow")
    ):
        os.makedirs(
            os.path.join(ROOT_DIR_PROJECT, "data", project_name, "models", "mlflow")
        )
