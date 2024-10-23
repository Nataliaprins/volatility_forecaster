from volatility_forecaster.forecast.forecast_n_steps import forecast_n_steps
from volatility_forecaster.forecast.forecast_one_step import forecast_one_step

# forecast one step
forecast_one_step_ahead = forecast_one_step(
    project_name="yahoo",
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/158973272966166127/197bedaf8fde4057bb4ea53b75e30f5f/artifacts/model",
    column_name="log_yield",
)
print(forecast_one_step_ahead)


forecast_n_steps_ahead = forecast_n_steps(
    project_name="yahoo",
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/158973272966166127/197bedaf8fde4057bb4ea53b75e30f5f/artifacts/model",
    column_name="log_yield",
    n_steps=5,
)
print(forecast_n_steps_ahead)
