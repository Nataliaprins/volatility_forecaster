from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from volatility_forecaster.constants import project_name
from volatility_forecaster.sklearn.make_experiment import make_experiment

model_instance = RandomForestRegressor()
param_dict = {
    "max_depth": [1, 2, 3, 4, 5],
    "criterion": ["friedman_mse", "absolute_error"],
}


make_experiment(
    model_type="sklearn",
    train_size=0.75,
    lags=3,
    model_instance=model_instance,
    param_dict=param_dict,
    root_dir=project_name,
    n_splits=5,
)
