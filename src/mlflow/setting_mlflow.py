"this fuction is used to log the model to mlflow"

from mlflow import MlflowClient
import mlflow.sklearn

import mlflow

def autologging_mlflow():
    # setting tracking directory
    artifact_location = Path.cwd().joinpath("data","yahoo","models","mlflow", "mlruns").as_uri()
    mlflow.set_tracking_uri(artifact_location)
    print('Tracking directory:', mlflow.get_tracking_uri())

    #autologging
    mlflow.sklearn.autolog(
        log_input_examples= False,
        log_model_signatures=True,
        log_models=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        max_tuning_runs=10,
        log_post_training_metrics=True,
        serialization_format= "cloudpickle",
        registered_model_name=None,
        )
