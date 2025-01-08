"this fuction is used to log the model to mlflow"

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from src.constants import ROOT_DIR_PROJECT, project_name


def autologging_mlflow(model_type: str = "sklearn"):

    # create the working directory
    if not os.path.exists(
        os.path.join(ROOT_DIR_PROJECT, "data", project_name, "models", "mlflow")
    ):
        os.makedirs(
            os.path.join(ROOT_DIR_PROJECT, "data", project_name, "models", "mlflow")
        )

    # setting tracking directory
    artifact_location = (
        Path.cwd().joinpath("data", "yahoo", "models", "mlflow", "mlruns").as_uri()
    )
    mlflow.set_tracking_uri(artifact_location)
    print("Tracking directory:", mlflow.get_tracking_uri())

    # autologging
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        max_tuning_runs=10,
        log_post_training_metrics=True,
        serialization_format="cloudpickle",
        registered_model_name=None,
    )
    return print("--MSG--autologging enabled for {}".format(model_type))


if __name__ == "__main__":
    autologging_mlflow(model_type="statsmodels")
