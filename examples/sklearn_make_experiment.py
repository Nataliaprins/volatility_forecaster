from sklearn.ensemble import RandomForestRegressor
from src.sklearn.make_experiments import make_experiments

model_instance = RandomForestRegressor()
param_dict = {
    "max_depth": [1, 2, 3, 4, 5],
    "criterion": ["friedman_mse", "absolute_error"],
}

project_name = "yahoo"

make_experiments(
    project_name=project_name,
    model_type="sklearn",
    column_name="log_yield",
    train_size=0.75,
    lags=3,
    model_instance=model_instance,
    param_dict=param_dict,
    n_splits=5,
)
