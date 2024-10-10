import os
from pathlib import Path

import mlflow

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name


def autologging(model_type: str = "ARCH"):

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
    mlflow.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
    )

    return print("--MSG--autologging enabled for {}".format(model_type))


if __name__ == "__main__":
    autologging(model_type="ARCH")
