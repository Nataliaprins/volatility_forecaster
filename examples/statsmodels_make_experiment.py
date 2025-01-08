from sklearn.model_selection import ParameterGrid
from src.statsmodels.make_experiments import make_experiments

param_dict = {
    "trend": ["c", "t"],
}
param_combinations = list(ParameterGrid(param_dict))
lags = 5
fit_params = {"cov_type": "HC0"}


make_experiments(
    project_name="yahoo",
    param_combinations=param_combinations,
    lags=lags,
    train_size=0.8,
    fit_params=fit_params,
)
