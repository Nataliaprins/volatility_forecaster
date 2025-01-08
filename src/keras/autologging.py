"this function sets the experiment and logs the results in mlflow for keras models."
import os
from pathlib import Path

import mlflow
import mlflow.keras
from src.constants import ROOT_DIR_PROJECT, project_name


def autologging_mlflow():

    # create the working directory
    create_mlflow_models_directory()

    # setting tracking directory
    artifact_location = (
        Path.cwd().joinpath("data", "yahoo", "models", "mlflow", "mlruns").as_uri()
    )
    print("Artifact location:", artifact_location)
    mlflow.set_tracking_uri(artifact_location)
    print("Tracking directory:", mlflow.get_tracking_uri())

    mlflow.tensorflow.autolog(
        log_models=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        registered_model_name=None,
    )

    print("--MSG--autologging enabled for keras")


# TODO: mover a core
def create_mlflow_models_directory():
    if not os.path.exists(
        os.path.join(
            ROOT_DIR_PROJECT, "data", project_name, "models", "mlflow", "mlruns"
        )
    ):
        os.makedirs(
            os.path.join(
                ROOT_DIR_PROJECT, "data", project_name, "models", "mlflow", "mlruns"
            )
        )
