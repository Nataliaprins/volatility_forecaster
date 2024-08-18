import keras_tuner as kt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from volatility_forecaster.pull_data.load_data import load_data

# Cargamos los datos
data = load_data(stock_name="googl", root_dir="yahoo")
returns = data["log_yield"]
returns = returns.dropna().values.reshape(-1, 1)
print(returns.shape)

# Normalizamos los datos
MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = MinMaxScaler.fit_transform(returns)
print(scaled_data.shape)


# creamos manaualmente las secuencias de tiempo
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


seq_length = 7
X, y = create_sequences(scaled_data, seq_length)
print(X.shape, y.shape)

# Dividimos los datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# redefinimos para que sean 3D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(X_train.shape, X_test.shape)


# Creamos el modelo
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


tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=10,
    factor=3,
    directory="my_dir",
    project_name="lstm_tuner",
)

# iniciaresmos la busqueda
tuner.search(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
)
# obtener el mejor modelo
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# evaluamos el modelo
test_loss = best_model.evaluate(X_test, y_test)
print(test_loss)

# Pronosticar los valores de prueba
predictions = best_model.predict(X_test)

# Desnormalizar las predicciones
predictions = MinMaxScaler.inverse_transform(predictions)
y_test_actual = MinMaxScaler.inverse_transform(y_test.reshape(-1, 1))

# Comparar resultados
import matplotlib.pyplot as plt

plt.plot(y_test_actual, color="blue", label="Actual Returns")
plt.plot(predictions, color="red", label="Predicted Returns")
plt.title("LSTM Model - Returns Prediction")
plt.xlabel("Time")
plt.ylabel("Returns")
plt.legend()
plt.show()
