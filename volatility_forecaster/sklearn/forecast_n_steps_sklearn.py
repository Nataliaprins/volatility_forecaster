"this function forecasts one step ahead using a logged model from mlflow."
import mlflow.sklearn

from volatility_forecaster.preprocessing.extract_serie import extract_serie
from volatility_forecaster.preprocessing.generate_lagged_data import (
    generate_lagged_data,
)


def forecast_n_steps_sklearn(
    project_name, stock_name, logged_model_path, n_steps, lags, column_name
):
    """
    this function forecasts n steps ahead using a logged model from mlflow. n_steps is the number of steps to forecast,
    lags should be greater than n_steps
    """
    if lags < n_steps:
        raise ValueError("--MSG-- n_steps should be less than lags")
    else:
        serie = extract_serie(
            stock_name=stock_name, project_name=project_name, column_name=column_name
        )
        # TODO: que tome los datos de los conjuntos de test (es la ultima info que se tiene)
        lagged_data = generate_lagged_data(
            project_name=project_name,
            stock_name=stock_name,
            serie=serie,
            lags=lags,
            column_name=column_name,
        ).dropna()

        x = lagged_data.drop(column_name, axis=1)

        model = mlflow.sklearn.load_model(logged_model_path)

        prediction = model.predict(x[-n_steps:])
        # TODO:pegar pronostico y meterlo en x

        return prediction


if __name__ == "__main__":
    forecast_n_steps_sklearn(
        project_name="yahoo",
        stock_name="googl",
        lags=3,
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/366621011459063161/10f78e028d604a618a9fdd4620e0ef7d/artifacts/best_estimator",
        n_steps=3,
        column_name="log_yield",
    )
