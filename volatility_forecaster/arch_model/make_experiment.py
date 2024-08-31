"this functions is used to make the experiment with mlflow and the arch model"

import os
from pathlib import Path

import mlflow
import mlflow.client
import pandas as pd
from arch import arch_model

from volatility_forecaster.arch_model._class_ArchModelWrapper import ArchModelWrapper
from volatility_forecaster.arch_model.autologging import autologging
from volatility_forecaster.arch_model.reshape_parameters import reshape_parameters
from volatility_forecaster.arch_model.simulate_data import simulate_data
from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name
from volatility_forecaster.core._extract_stock_name import _extract_stock_name
from volatility_forecaster.core._get_data_files import _get_data_files
from volatility_forecaster.core.arch import train_test_split
from volatility_forecaster.metrics.evaluate_models import evaluate_models
from volatility_forecaster.pull_data import load_data


def make_experiment(
    param_combinations,
    train_size,
    fit_params_combinations,
):

    data_files = _get_data_files()

    for data_file in data_files:
        stock_name = _extract_stock_name(data_file)

        data = load_data.load_data(stock_name, project_name)
        returns = data["log_yield"]
        returns = returns.dropna()

        train, test = train_test_split.train_test_split(returns, train_size)

        # TODO: extraer metodo
        param_combinations_str = {
            key: str(value) for key, value in param_combinations.items()
        }
        run_name = "_".join(
            [f"{key}:{value}" for key, value in param_combinations_str.items()]
        )
        model_type = param_combinations["vol"]

        create_mlflow_directories()
        tracking_location = set_mlflow_tracking_uri()

        mlflow.set_experiment(stock_name)

        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName", run_name)

            # get run info

            model = ArchModelWrapper(**param_combinations)
            res = model.fit(train, fit_params_combinations)

            # log params
            if param_combinations:
                for param, value in param_combinations.items():
                    mlflow.log_param(param, value)

            # log fit params
            if fit_params_combinations:
                for param, value in fit_params_combinations.items():
                    mlflow.log_param(param, value)

            # predict
            y_pred = model.predict(None, train[:-1])
            y_true = train[-len(y_pred) :]

            # log metrics
            metrics = evaluate_models(y_true, y_pred)
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            run = mlflow.active_run()
            artifact_uri = run.info.artifact_uri
            model.save_model(artifact_uri)
            mlflow.log_artifact(
                os.path.join(artifact_uri.replace("file://", ""), "arch_model.pkl")
            )
            # log model
            log_arch_model(model)


def log_arch_model(model):
    mlflow.pyfunc.log_model(
        artifact_path="artifacts",
        python_model=model,
        conda_env=None,
        code_path=[],
    )


def create_mlflow_directories():
    if not os.path.exists(
        os.path.join(ROOT_DIR_PROJECT, "data", project_name, "models", "mlflow")
    ):
        os.makedirs(
            os.path.join(ROOT_DIR_PROJECT, "data", project_name, "models", "mlflow")
        )


def set_mlflow_tracking_uri():
    artifact_location = (
        Path.cwd().joinpath("data", "yahoo", "models", "mlflow", "mlruns").as_uri()
    )
    mlflow.set_tracking_uri(artifact_location)
    return mlflow.get_tracking_uri()
