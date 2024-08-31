import mlflow
from arch import arch_model


# log model
def log_arch_model(arch_wrapper, model_name):
    with mlflow.start_run():
        # Guardar el modelo con el flavor pyfunc
        mlflow.pyfunc.log_model(
            artifact_path="arch_model",
            python_model=arch_wrapper,
            registered_model_name=model_name,
        )
