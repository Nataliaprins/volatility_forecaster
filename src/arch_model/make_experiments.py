from src.arch_model.make_experiment import make_experiment


def make_experiments(
    project_name,
    param_combinations,
    train_size,
    fit_params_combinations,
):
    for combination in param_combinations:
        for fit_params_combination in fit_params_combinations:
            make_experiment(
                project_name=project_name,
                param_combinations=combination,
                train_size=train_size,
                fit_params_combinations=fit_params_combination,
            )
