from kerastuner import RandomSearch

from examples.keras_package import build_model
from volatility_forecaster.keras.train_test_split import train_test_split


def tuning_params(num_trials):
    tuner = RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=num_trials,
        executions_per_trial=1,
        project_name="lstm_tuner",
    )
    print(tuner.search_space_summary())
    return tuner.get_best_hyperparameters(num_trials=num_trials)[0]


if __name__ == "__main__":
    build_model(tuning_params(5))
