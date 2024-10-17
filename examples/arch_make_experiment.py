from sklearn.model_selection import ParameterGrid

from volatility_forecaster.arch_model.make_experiments import make_experiments

project_dir = "yahoo"

param_dict = (
    {
        "p": [1, 2],
        "q": [1, 2, 3],
        "vol": ["GARCH"],
        "dist": ["skewt"],
    },
)
param_combinations = list(ParameterGrid(param_dict))

fit_params = {"update_freq": [5, 6, 7], "disp": ["off"]}
fit_params_combinations = list(ParameterGrid(fit_params))

forecast_params = {"horizon": 2, "method": "analytic"}

make_experiments(
    project_name=project_dir,
    param_combinations=param_combinations,
    train_size=0.8,
    fit_params_combinations=fit_params_combinations,
)

# forecast one step
forecast_one_step_ahead = forecast_one_step(
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/449779083138429514/5c7d7c596dd54552911df045b2a55300/artifacts/artifacts",
    column_name="log_yield",
)
print(forecast_one_step_ahead)

# forecast n steps
forecast_n_steps_ahead = forecast_n_steps(
    stock_name="googl",
    logged_model_path="file:///Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/449779083138429514/10792a8e5d0341e493f4c5905f459963/artifacts/best_estimator",
    column_name="log_yield",
    n_steps=5,
)
