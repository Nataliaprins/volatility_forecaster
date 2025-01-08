from src.core.directories.create_reports_dir import create_reports_dir
from src.core.forecast.n_step_forecast import n_step_forecast
from src.core.forecast.one_step_forecast import one_step_forecast
from src.reports.save_forecast import save_forecast

project_name = "yahoo"
logged_model_path = "file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/695742625896702102/89f5b9a9885143a0bf3f458838bf5eef/artifacts/model"
model_type = "keras"
stock_name = "aapl"
column_name = "log_yield"
n_steps = 10

create_reports_dir(project_name=project_name)

forecast_one_step_ahead = one_step_forecast(
    model_type=model_type,
    project_name=project_name,
    stock_name=stock_name,
    logged_model_path=logged_model_path,
    column_name=column_name,
    lags=1,
)
print(forecast_one_step_ahead)

forecast_n_step_ahead = n_step_forecast(
    model_type=model_type,
    project_name=project_name,
    stock_name=stock_name,
    logged_model_path=logged_model_path,
    column_name=column_name,
    n_steps=n_steps,
    lags=3,
)
print(forecast_n_step_ahead)

save_forecast(
    project_name="yahoo",
    model_type="arch",
    stock_name="aapl",
    forecast_n_step_ahead=forecast_n_step_ahead,
)
