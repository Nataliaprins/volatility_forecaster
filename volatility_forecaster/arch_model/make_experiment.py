"this functions is used to make the experiment with mlflow and the arch model"
import glob
import os

import mlflow
from arch import arch_model
from arch.univariate import GARCH
from mlflow import log_artifact, log_metric, log_param
from sklearn.metrics import mean_squared_error

from volatility_forecaster.arch_model.fit import fit_model
from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name
from volatility_forecaster.core.arch import train_test_split
from volatility_forecaster.mlflow.setting_mlflow import autologging_mlflow
from volatility_forecaster.pull_data import load_data


def make_experiment(
    param_combinations,
    train_size,
    fit_params,
    forecast_params,
):

    # obtain the data path
    path_to_data = os.path.join(
        ROOT_DIR_PROJECT,
        "data",
        project_name,
        "processed",
        "prices",
        "*.csv",
    )

    data_files = glob.glob(path_to_data)

    for data_file in data_files:
        # define the stock name
        # the data_file contains the path like yahoo/processed/prices/AAPL.csv
        stock_name_with_ext = os.path.basename(data_file)
        stock_name, _ = os.path.splitext(stock_name_with_ext)

        # Load the data
        data = load_data.load_data(stock_name, project_name)
        # obtain Y (retuns)
        returns = data["log_yield"]
        returns = returns.dropna()

        # train test split
        train, test = train_test_split.train_test_split(returns, train_size)

        # Set the experiment
        param_combinations_str = {
            key: str(value) for key, value in param_combinations.items()
        }
        model_type = "ARCH" + repr(param_combinations_str)
        autologging_mlflow(model_type=model_type)

        # set the experiment if it exists
        mlflow.set_experiment(stock_name)

        # start the experiment
        with mlflow.start_run():
            # fit the model
            model = arch_model(train, **param_combinations)
            res = model.fit(**fit_params)

            # forecast
            forecast = res.forecast(**forecast_params)

            y_pred = forecast.variance.tail()
            print(y_pred)

        # TODO: log metrics and params, set the name of run
        # log the metrics


if __name__ == "__main__":
    make_experiment(
        param_combinations={"p": 1, "q": 1, "vol": "GARCH"},
        train_size=0.8,
        fit_params={"disp": "off"},
        forecast_params={"horizon": 2, "method": "analytic"},
    )
