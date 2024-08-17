"import the necessary packages"
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from volatility_forecaster.keras.make_experiment import make_experiment

#
seq_length = 7  # sequence length to create inputs and outputs
scaler_instance = "min_max"
scaler_params = {"feature_range": (0, 1)}


# define the keras model using the hyperparameters
def build_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            units=hp.Int("units", min_value=32, max_value=256, step=32),
            return_sequences=False,
            input_shape=(seq_length, 1),
        )
    )
    model.add(Dense(1, activation="sigmoid"))
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG"
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


num_trials = 3
model_name = "lstm"

# call make_experiment
make_experiment(
    model_name=model_name,
    model=build_model,
    scaler_instance=scaler_instance,
    scaler_params=scaler_params,
    seq_length=seq_length,
    train_size=0.7,
    num_trials=num_trials,
)
