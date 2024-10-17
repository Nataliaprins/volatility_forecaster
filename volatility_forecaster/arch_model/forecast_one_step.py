import mlflow.pyfunc

from volatility_forecaster.preprocessing.extract_serie import extract_serie


def forecast_one_step(stock_name, logged_model_path, column_name):
    data = extract_serie(stock_name, project_name="yahoo", column_name=column_name)
    loaded_model = mlflow.pyfunc.load_model(logged_model_path)
    prediction = loaded_model.predict(data)
    return prediction


if __name__ == "__main__":
    forecast_one_step(
        stock_name="googl",
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/449779083138429514/5c7d7c596dd54552911df045b2a55300/artifacts/artifacts",
        column_name="log_yield",
    )
