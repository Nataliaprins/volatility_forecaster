from volatility_forecaster.sklearn.make_experiment import make_experiment


def make_experiments(
    model_type,
    train_size,
    lags,
    model_instance,
    model_params,
    root_dir,
    n_splits,
):
    for combination in param_combinations:
        make_experiment(
            root_dir=root_dir,
            parameters=combination,
            train_size=train_size,
            fit_params=fit_params,
            forecast_params=forecast_params,
        )
