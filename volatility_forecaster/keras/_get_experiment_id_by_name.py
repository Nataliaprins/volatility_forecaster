import mlflow


def _get_experiment_id_by_name(stock_name):
    experiment = mlflow.get_experiment_by_name(str(stock_name))
    if experiment is not None:
        return experiment.experiment_id
    else:
        # create the experiment
        raise ValueError(f"Experiment with name '{stock_name}' does not exist.")
