import mlflow
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

from volatility_forecaster.core._extract_stock_name import _extract_stock_name
from volatility_forecaster.core._get_data_files import _get_data_files
from volatility_forecaster.core.mlflow.log_statsmodels_model import (
    log_statsmodels_model,
)
from volatility_forecaster.core.statsmodels.generate_parameter_string import (
    generate_parameter_string,
)
from volatility_forecaster.core.statsmodels.train_test_split import train_test_split
from volatility_forecaster.metrics.evaluate_models import evaluate_models
from volatility_forecaster.pull_data import load_data


def make_experiment(
    project_name,
    train_size,
    lags,
    param_combinations,
    fit_params,
):

    data_files = _get_data_files(project_name=project_name)

    for data_file in data_files:
        stock_name = _extract_stock_name(data_file)

        data = load_data.load_data(stock_name, project_name)

        returns = data["log_yield"].dropna()

        train, test = train_test_split(
            data=returns,
            project_name=project_name,
            stock_name=stock_name,
            ratio=train_size,
        )

        log_statsmodels_model(project_name=project_name)
        mlflow.set_experiment(str(stock_name))

        parameters = generate_parameter_string(param_combinations)
        run_name = f"Autoregresive_lags:{lags}_{parameters}"

        with mlflow.start_run() as run:
            mlflow.set_tag("mlflow.runName", run_name)
            model = AutoReg(train, lags=lags, **param_combinations)
            results = model.fit(**fit_params)
            # predict
            ypred = results.predict(start=len(train), end=len(train) + len(test) - 1)
            # log parameters
            mlflow.log_params(param_combinations)
            mlflow.log_param("lags", lags)

            # log metrics
            metrics = evaluate_models(test, ypred)
            mse = metrics["mse"]
            mae = metrics["mae"]
            mlflow.log_metrics({"mse": mse, "mae": mae})
