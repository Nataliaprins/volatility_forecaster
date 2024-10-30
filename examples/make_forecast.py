from volatility_forecaster.core.forecast.n_step_forecast import n_step_forecast
from volatility_forecaster.core.forecast.one_step_forecast import one_step_forecast

forecast_one_step_ahead = one_step_forecast(
    model_type="keras",
    project_name="yahoo",
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/446645761122062445/d3e81d982a994eb8a92304431dd8add2/artifacts/model",
    column_name="log_yield",
)
print(forecast_one_step_ahead)


forecast_n_steps_ahead = n_step_forecast(
    model_type="keras",
    project_name="yahoo",
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/446645761122062445/d3e81d982a994eb8a92304431dd8add2/artifacts/model",
    column_name="log_yield",
    n_steps=5,
)
print(forecast_n_steps_ahead)
