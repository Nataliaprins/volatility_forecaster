from volatility_forecaster.statsmodels.make_experiment import make_experiment


def make_experiments(
    project_name,
    param_combinations,
    lags,
    train_size,
    fit_params,
):
    for combination in param_combinations:
        make_experiment(
            project_name=project_name,
            param_combinations=combination,
            lags=lags,
            train_size=train_size,
            fit_params=fit_params,
        )
