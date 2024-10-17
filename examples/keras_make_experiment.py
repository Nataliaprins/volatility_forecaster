"import the necessary packages"
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from volatility_forecaster.keras.make_experiments import make_experiments

# define the directory where the data is stored, named in prepare_project.py
project_name = "yahoo"

seq_length = 7  # sequence length to create inputs and outputs
scaler_instance = "min_max"
scaler_params = {"feature_range": (0, 1)}


# define the keras model using the hyperparameters
def build_model(hp):
    model = Sequential()
    # LSTM layers
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            LSTM(
                units=hp.Int(f"units_{i}", min_value=50, max_value=150, step=50),
                return_sequences=(i < hp.Int("num_layers", 1, 3) - 1),
                input_shape=(seq_length, 1),
            )
        )
        model.add(
            Dropout(
                hp.Float(
                    f"dropout_rate_{i}", min_value=0.002, max_value=0.05, step=0.0001
                )
            )
        )

    # Dense layer
    model.add(Dense(1))
    # Compile model
    model.compile(
        optimizer=hp.Choice("optimizer", ["adam", "rmsprop", "sgd"]),
        loss="mean_squared_error",
    )

    return model


num_max_epochs = 10
model_name = "lstm"

# call make_experiment
make_experiments(
    project_name=project_name,
    model_name=model_name,
    model=build_model,
    scaler_instance=scaler_instance,
    scaler_params=scaler_params,
    seq_length=seq_length,
    train_size=0.7,
    num_max_epochs=num_max_epochs,
)
