" this function makes runs the statisticts models with mlflow tracking and using Stastsmodels library"

import glob
import os

import mlflow
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name
from volatility_forecaster.core.statsmodels.train_test_split import train_test_split
from volatility_forecaster.metrics.evaluate_models import evaluate_models
from volatility_forecaster.mlflow.setting_mlflow import autologging_mlflow
from volatility_forecaster.pull_data import load_data

# TODO: insertar variable model_type. revisar no corre


def make_experiment_statsmodels(
    train_size,
    lags,
    param_combinations,
    fit_params,
    model_type,
):

    # obtain the data path
    path_to_data = os.path.join(
        ROOT_DIR_PROJECT, "data", project_name, "processed", "prices", "*.csv"
    )
    data_files = glob.glob(path_to_data)

    # iterate over the data files
    for data_file in data_files:
        # define the stock name
        stock_name_with_ext = os.path.basename(data_file)
        stock_name, _ = os.path.splitext(stock_name_with_ext)

        # Load the data
        data = load_data.load_data(stock_name, project_name)
        returns = data["log_yield"].dropna()

        # train test split
        train, test = train_test_split(returns, train_size)

        # auto logging
        autologging_mlflow(model_type="statsmodels")

        mlflow.set_experiment(str(stock_name))

        # start the experiment
        with mlflow.start_run(run_name="statsmodels_" + stock_name) as run:
            # fit the model passed in the argument model_instance
            model = AutoReg(train, lags=lags, **param_combinations)
            results = model.fit(**fit_params)

    print(f"--MSG: Experiment finished for {model_type}--")


if __name__ == "__main__":
    make_experiment_statsmodels(
        root_dir=project_name,
        train_size=0.8,
        lags=5,
        param_combinations={},
        model_type="statsmodels",
    )
