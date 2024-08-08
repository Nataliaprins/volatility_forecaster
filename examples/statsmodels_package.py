from sklearn.model_selection import ParameterGrid

from volatility_forecaster.statsmodels.make_experiments import make_experiments

param_dict = {
    "trend": ["c", "t"],
}
param_combinations = list(ParameterGrid(param_dict))
lags = 5
fit_params = {"cov_type": "HC0"}


make_experiments(
    param_combinations=param_combinations,
    lags=lags,
    model_type="AR",
    train_size=0.8,
    fit_params=fit_params,
)
