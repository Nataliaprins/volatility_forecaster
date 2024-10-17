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
