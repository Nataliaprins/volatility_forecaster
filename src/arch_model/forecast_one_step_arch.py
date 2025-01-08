import mlflow.pyfunc
from src.preprocessing.extract_serie import extract_serie


def forecast_one_step_arch(
    project_name,
    stock_name,
    logged_model_path,
    column_name,
):
    data = extract_serie(stock_name, project_name=project_name, column_name=column_name)
    loaded_model = mlflow.pyfunc.load_model(logged_model_path)
    model_input = {"data": data, "horizon": 1}
    prediction = loaded_model.predict(model_input)
    return prediction


if __name__ == "__main__":
    forecast_one_step_arch(
        project_name="yahoo",
        stock_name="aapl",
        logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/490905807558364708/0f0bf6a1285544859dce1ab9206c94fb/artifacts/artifacts",
        column_name="log_yield",
    )
