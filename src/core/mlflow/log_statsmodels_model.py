import os
from pathlib import Path

import mlflow
import mlflow.statsmodels
import statsmodels.api as sm
from src.constants import ROOT_DIR_PROJECT
from src.core.mlflow.create_mlflow_directories import create_mlflow_directories
from src.core.mlflow.set_mlflow_tracking_uri import set_mlflow_tracking_uri


def log_statsmodels_model(project_name: str):

    create_mlflow_directories(project_name)
    set_mlflow_tracking_uri(project_name)

    mlflow.statsmodels.autolog(
        log_models=True,
        log_datasets=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        registered_model_name=None,
    )

    print("--MSG--autologging enabled for statsmodels")


if __name__ == "__main__":
    log_statsmodels_model(project_name="yahoo")
