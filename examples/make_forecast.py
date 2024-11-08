from volatility_forecaster.core.directories.create_reports_dir import create_reports_dir
from volatility_forecaster.core.forecast.n_step_forecast import n_step_forecast
from volatility_forecaster.core.forecast.one_step_forecast import one_step_forecast
from volatility_forecaster.reports.save_forecast import save_forecast

create_reports_dir(project_name="yahoo")

# forecasting one step ahead and n steps ahead. Only forecast_n_step_ahead is saved in a csv file

forecast_one_step_ahead = one_step_forecast(
    model_type="arch",
    project_name="yahoo",
    stock_name="aapl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/490905807558364708/0f0bf6a1285544859dce1ab9206c94fb/artifacts/artifacts",
    column_name="log_yield",
    lags=1,
)
print(forecast_one_step_ahead)

forecast_n_step_ahead = n_step_forecast(
    model_type="arch",
    project_name="yahoo",
    stock_name="aapl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/490905807558364708/0f0bf6a1285544859dce1ab9206c94fb/artifacts/artifacts",
    column_name="log_yield",
    n_steps=10,
    lags=3,
)
print(forecast_n_step_ahead)

save_forecast(
    project_name="yahoo",
    model_type="arch",
    stock_name="aapl",
    forecast_n_step_ahead=forecast_n_step_ahead,
)
