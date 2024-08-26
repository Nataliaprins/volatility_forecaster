"this functions is used to make the experiment with mlflow and the arch model"

import mlflow
import pandas as pd
from arch import arch_model

from volatility_forecaster.arch_model.autologging import autologging
from volatility_forecaster.arch_model.reshape_parameters import reshape_parameters
from volatility_forecaster.arch_model.simulate_data import simulate_data
from volatility_forecaster.constants import project_name
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

        param_combinations_str = {
            key: str(value) for key, value in param_combinations.items()
        }
        run_name = "_".join(
            [f"{key}:{value}" for key, value in param_combinations_str.items()]
        )
        model_type = param_combinations["vol"]

        autologging(model_type=model_type)

        mlflow.set_experiment(stock_name)

        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName", run_name)
            model = arch_model(train, **param_combinations)
            res = model.fit(**fit_params_combinations)

            sim_parameters = pd.DataFrame(res.params)
            sim_parameters = reshape_parameters(sim_parameters)

            sim_data = simulate_data(
                param_combinations=param_combinations,
                sim_parameters=sim_parameters,
                nobs=len(test),
            )

            y_pred = sim_data["volatility"]
            metrics = evaluate_models(test, y_pred)
            mae = metrics["mae"]
            mse = metrics["mse"]

            # mlflow.log_params(**param_combinations)
            # mlflow.log_params(**fit_params_combinations)
            mlflow.log_metrics({"mae": mae, "mse": mse})
