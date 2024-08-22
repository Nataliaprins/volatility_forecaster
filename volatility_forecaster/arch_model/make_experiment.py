"this functions is used to make the experiment with mlflow and the arch model"

import mlflow
from arch import arch_model

from volatility_forecaster.constants import project_name
from volatility_forecaster.core._extract_stock_name import _extract_stock_name
from volatility_forecaster.core._get_data_files import _get_data_files
from volatility_forecaster.core.arch import train_test_split
from volatility_forecaster.mlflow.setting_mlflow import autologging_mlflow
from volatility_forecaster.pull_data import load_data


def make_experiment(
    param_combinations,
    train_size,
    fit_params,
    forecast_params,
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
        model_type = "ARCH" + repr(param_combinations_str)
        autologging_mlflow(model_type=model_type)

        mlflow.set_experiment(stock_name)

        with mlflow.start_run():
            model = arch_model(train, **param_combinations)
            res = model.fit(**fit_params)
            forecast = res.forecast(**forecast_params)
            y_pred = forecast.variance.tail()
            print(y_pred)

        # TODO: log metrics and params, set the name of run
        # log the metrics
