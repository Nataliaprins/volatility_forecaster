"this functions is used to make the experiment with mlflow and the arch model"

import os

import mlflow
import mlflow.client

from volatility_forecaster.arch_model.log_arch_model import log_arch_model
from volatility_forecaster.core._extract_stock_name import _extract_stock_name
from volatility_forecaster.core._get_data_files import _get_data_files
from volatility_forecaster.core.arch import train_test_split
from volatility_forecaster.core.arch._class_ArchModelWrapper import ArchModelWrapper
from volatility_forecaster.core.arch.generate_run_name import generate_run_name
from volatility_forecaster.core.arch.param_combinations_to_string import (
    param_combinations_to_string,
)
from volatility_forecaster.core.mlflow.create_mlflow_directories import (
    create_mlflow_directories,
)
from volatility_forecaster.core.mlflow.set_mlflow_tracking_uri import (
    set_mlflow_tracking_uri,
)
from volatility_forecaster.metrics.evaluate_models import evaluate_models
from volatility_forecaster.pull_data import load_data


def make_experiment(
    project_name,
    param_combinations,
    train_size,
    fit_params_combinations,
):

    data_files = _get_data_files(project_name=project_name)

    for data_file in data_files:
        stock_name = _extract_stock_name(data_file)

        data = load_data.load_data(stock_name, project_name)
        returns = data["log_yield"]
        returns = returns.dropna()

        train, test = train_test_split.train_test_split(returns, train_size)

        param_combinations_str = param_combinations_to_string(param_combinations)
        run_name = generate_run_name(param_combinations_str)

        create_mlflow_directories()

        set_mlflow_tracking_uri()

        mlflow.set_experiment(stock_name)

        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName", run_name)

            # get run info
            model = ArchModelWrapper(**param_combinations)
            results = model.fit(train, fit_params_combinations)

            # log params
            if param_combinations:
                for param, value in param_combinations.items():
                    mlflow.log_param(param, value)

            # log fit params
            if fit_params_combinations:
                for param, value in fit_params_combinations.items():
                    mlflow.log_param(param, value)

            # predict
            y_pred = model.predict(None, train)
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

    return print("--MSG: Experiment finished for ARCH model--")
