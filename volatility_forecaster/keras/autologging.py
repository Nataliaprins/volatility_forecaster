"this function sets the experiment and logs the results in mlflow for keras models."
import os
from pathlib import Path

import mlflow
import mlflow.keras

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name


def autologging_mlflow():

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

    mlflow.tensorflow.autolog(
        log_models=True,
        log_every_epoch=True,
        log_input_examples=False,
        log_model_signatures=True,
        log_datasets=True,
    )

    print("--MSG--autologging enabled for keras")
