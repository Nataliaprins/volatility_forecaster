from pathlib import Path

import mlflow


def set_mlflow_tracking_uri(project_name: str):
    artifact_location = (
        Path.cwd().joinpath("data", project_name, "models", "mlflow", "mlruns").as_uri()
    )
    mlflow.set_tracking_uri(artifact_location)
    print("Tracking directory:", mlflow.get_tracking_uri())
    return mlflow.get_tracking_uri()


if __name__ == "__main__":
    set_mlflow_tracking_uri(project_name="yahoo")
