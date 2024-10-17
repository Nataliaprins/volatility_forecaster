from volatility_forecaster.arch_model.forecast_n_steps import forecast_n_steps
from volatility_forecaster.arch_model.forecast_one_step import forecast_one_step

project_name = "yahoo"

# TODO: insert the project_name variable in the function call

# forecast one step
forecast_one_step_ahead = forecast_one_step(
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/822366107804400590/fa3b05ae93fb48ae8ff39d92eaccd1d5/artifacts/artifacts",
    column_name="log_yield",
)
print(forecast_one_step_ahead)


forecast_n_steps_ahead = forecast_n_steps(
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/449779083138429514/10792a8e5d0341e493f4c5905f459963/artifacts/best_estimator",
    column_name="log_yield",
    n_steps=5,
)
print(forecast_n_steps_ahead)
