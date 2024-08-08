from volatility_forecaster.statsmodels.make_experiment import (
    make_experiment_statsmodels,
)


def make_experiments(
    param_combinations,
    lags,
    model_type,
    train_size,
    fit_params,
):
    for combination in param_combinations:
        make_experiment_statsmodels(
            param_combinations=combination,
            lags=lags,
            model_type=model_type,
            train_size=train_size,
            fit_params=fit_params,
        )
