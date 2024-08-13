"import the necessary packages"
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from volatility_forecaster.keras.make_experiment import make_experiment

# from volatility_forecaster.keras.tuning_params import tuning_params

seq_length = 7  # sequence length to create inputs and outputs
scaler_instance = "min_max"
scaler_params = {"feature_range": (0, 1)}


# define the LSTM model using keras
def build_model(hp):
    model = Sequential()
    # TODO: agregar antes de la capa LSTM una capa temporal atetion para agregar pesos a las secuencias
    model.add(
        LSTM(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            input_shape=(seq_length, 1),
        )
    )
    model.add(Dropout(hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))

    # model.add(BatchNormalization())
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-6, max_value=1e-2, sampling="LOG"
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    print(model.summary())
    return model


# num_trials = 10 # number of trials for the hyperparameter search
# param_dict = tuning_params["lstm"] # list of hyperparameters to tune

# call make_experiment
make_experiment(
    model=build_model,
    scaler_instance=scaler_instance,
    scaler_params=scaler_params,
    seq_length=seq_length,
    train_size=0.7,
)
