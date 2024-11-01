from volatility_forecaster.core.directories.create_reports_dir import create_reports_dir
from volatility_forecaster.core.forecast.n_step_forecast import n_step_forecast
from volatility_forecaster.core.forecast.one_step_forecast import one_step_forecast

create_reports_dir(project_name="yahoo")

# forecasting one step ahead and n steps ahead. Only forecast_n_step_ahead is saved in a csv file

forecast_one_step_ahead = one_step_forecast(
    model_type="statsmodels",
    project_name="yahoo",
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/366621011459063161/8cdf99170e8a47b1993864b931647c32/artifacts/model",
    column_name="log_yield",
    lags=3,
)
print(forecast_one_step_ahead)
forecast_n_step_ahead = n_step_forecast(
    model_type="statsmodels",
    project_name="yahoo",
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/366621011459063161/8cdf99170e8a47b1993864b931647c32/artifacts/model",
    column_name="log_yield",
    n_steps=3,
    lags=3,
)
print(forecast_n_step_ahead)
