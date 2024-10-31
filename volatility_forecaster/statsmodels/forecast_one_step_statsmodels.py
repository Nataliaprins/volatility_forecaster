import os

import mlflow.statsmodels
import pandas as pd


def forecast_one_step_statsmodels(
    stock_name,
    project_name,
    logged_model_path,
):
    # TODO: extrer método para cargar datos de test meter a core
    test_data = pd.read_csv(
        os.path.join(
            "/Users/nataliaacevedo/volatility_forecaster/",
            "data",
            project_name,
            "processed",
            "train_test",
            "statsmodels",
            f"statsmodels_{stock_name}_test.csv",
        )
    )

    loaded_model = mlflow.statsmodels.load_model(logged_model_path)

    prediction = loaded_model.forecast(steps=(len(test_data) + 1))

    return prediction.tail(1)


if __name__ == "__main__":
    forecast_one_step_statsmodels(
        stock_name="googl",
        project_name="yahoo",
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/366621011459063161/8cdf99170e8a47b1993864b931647c32/artifacts/model",
        column_name="log_yield",
        lags=5,
    )
