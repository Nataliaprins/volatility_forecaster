from kerastuner import RandomSearch


def tuning_params(model, num_trials, model_name):
    tuner = RandomSearch(
        model,
        objective="val_loss",
        max_trials=num_trials,
        executions_per_trial=1,
        project_name=str(model_name),
    )
    return tuner
