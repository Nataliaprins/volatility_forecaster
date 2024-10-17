"this function is used to make an experiment using mlflow tracking for sk-learn models with the given parameters and choosing the best estimator."

import mlflow
import mlflow.sklearn

from volatility_forecaster.metrics.evaluate_models import evaluate_models
from volatility_forecaster.mlflow.fetch_logged_data import fetch_logged_data
from volatility_forecaster.mlflow.setting_mlflow import autologging_mlflow
from volatility_forecaster.sklearn.selector import selector
from volatility_forecaster.train_test_split.ts_train_test_split import (
    ts_train_test_split,
)


def make_experiment(
    project_name,
    stock_name,
    model_type,
    column_name,
    train_size,
    prod_size,
    lags,
    model_instance,
    param_dict,
    n_splits,
):

    x, y, x_train, x_test, y_train, y_test = ts_train_test_split(
        project_name=project_name,
        train_size=train_size,
        lags=lags,
        stock_name=stock_name,
        column_name=column_name,
        n_splits=n_splits,
    )

    estimator = selector(
        model_instance=model_instance,
        param_dict=param_dict,
        n_splits=n_splits,
        x=x,
        train_size=train_size,
    )

    autologging_mlflow(model_type=model_type)

    parameters = estimator.get_params()

    run_name = f"{repr(model_type)}_"

    mlflow.set_experiment(str(stock_name))

    with mlflow.start_run() as run:

        estimator.fit(x, y)
        best_model = estimator.best_estimator_
        y_pred = best_model.predict(x_test)

        metrics = evaluate_models(y_test, y_pred)

        mlflow.log_metric("mse", metrics["mse"])
        mlflow.log_metric("mae", metrics["mae"])

        best_params = estimator.best_params_

        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        print(params, tags, artifacts)

        mlflow.set_tag(
            "mlflow.runName",
            f"model: {repr(model_instance)} Run with params: {str(best_params)}",
        )
        print(f"--MSG: Experiment finished for {model_type}--")
