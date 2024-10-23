import mlflow.pyfunc

from volatility_forecaster.preprocessing.extract_serie import extract_serie


def forecast_one_step(
    project_name,
    stock_name,
    logged_model_path,
    column_name,
):
    data = extract_serie(stock_name, project_name=project_name, column_name=column_name)
    loaded_model = mlflow.pyfunc.load_model(logged_model_path)
    prediction = loaded_model.predict(data)
    return prediction


if __name__ == "__main__":
    forecast_one_step(
        project_name="yahoo",
        stock_name="googl",
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/158973272966166127/e163c8aadf5645c59e9847f23cbbe20f/artifacts/model",
        column_name="log_yield",
    )
