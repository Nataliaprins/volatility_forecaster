from pathlib import Path

import mlflow


def set_mlflow_tracking_uri():
    artifact_location = (
        Path.cwd().joinpath("data", "yahoo", "models", "mlflow", "mlruns").as_uri()
    )
    mlflow.set_tracking_uri(artifact_location)
    return mlflow.get_tracking_uri()
