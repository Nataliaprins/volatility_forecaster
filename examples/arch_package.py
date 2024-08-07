from sklearn.model_selection import ParameterGrid

from volatility_forecaster.arch_model.make_experiments import make_experiments

param_dict = (
    {
        "p": [1, 2],
        "q": [1, 2, 3],
        "vol": ["GARCH", "ARCH"],
    },
)
param_combinations = list(ParameterGrid(param_dict))

fit_params = {"disp": "off"}

forecast_params = {"horizon": 2, "method": "analytic"}

make_experiments(
    param_combinations=param_combinations,
    train_size=0.8,
    fit_params=fit_params,
    forecast_params=forecast_params,
)
