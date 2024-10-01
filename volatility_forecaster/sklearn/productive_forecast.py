"this function forecasts one step ahead using a logged model from mlflow."
import mlflow.pyfunc

from volatility_forecaster.preprocessing.extract_serie import extract_serie
from volatility_forecaster.preprocessing.generate_lagged_data import (
    generate_lagged_data,
)
from volatility_forecaster.train_test_split.extract_production_data import (
    extract_production_data,
)


def productive_forecast(stock_name, logged_model_path, lags, prod_size, column_name):
    serie = extract_serie(stock_name, column_name)
    lagged_df = generate_lagged_data(lags, serie).dropna()
    productive_data = extract_production_data(prod_size, lagged_df)

    x = productive_data.drop(columns=["log_yield"])

    model = mlflow.pyfunc.load_model(logged_model_path)
    y_pred = model.predict(x)
    return y_pred


if __name__ == "__main__":
    productive_forecast(
        stock_name="googl",
        lags=3,
        prod_size=0.1,
        logged_model_path="/Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/139047356004469501/ef234625915d4b40bb6e85ab94bc81e3/artifacts/best_estimator",
        column_name="log_yield",
    )
