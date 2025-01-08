import mlflow.pyfunc


def log_arch_model(model):
    mlflow.pyfunc.log_model(
        artifact_path="artifacts",
        python_model=model,
        conda_env=None,
        code_paths=[],
    )
