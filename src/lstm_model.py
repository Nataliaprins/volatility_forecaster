"this function is used to create the lstm model using keras tunnig for optimal hyperparameters"
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from constants import ROOT_DIR_PROJECT
from create_sequences import create_sequences
from load_data import load_data


def lstm_model(  root_dir, seq_length, train_size, epoch):
    # obtain the data path 
    path_to_data = os.path.join(ROOT_DIR_PROJECT, "yahoo", "processed", "prices", "*.csv")
    data_files = glob.glob(path_to_data)
    stock_names = [Path(file).stem.split("_")[1] for file in data_files]

    for stock in stock_names:
        df = load_data(stock_name=stock, root_dir=root_dir)
        df = df.dropna()
        serie= df["rolling_std"]

        #scale the data with min max scaler
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #scaled_data = scaler.fit_transform(serie.values.reshape(-1, 1))

        #create sequences
        X, y = create_sequences(serie, seq_length)
        print(X.shape, y.shape)

        # define the LSTM model using keras and keras tuner
        def build_model(hp):
            model = Sequential()
            #agregar antes de la capa LSTM una capa temporal atetion para agregar pesos a las secuencias
            model.add(LSTM(units=hp.Int("units", 
                                        min_value=32, max_value=512, step=32), 
                                        input_shape=(seq_length, 1)))
            model.add(Dense(1))
            model.add(Dropout(hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)))
            model.add(BatchNormalization())
            learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
            return model    

        tuner= RandomSearch(
            build_model, 
            objective="val_loss", 
            max_trials=3, 
            executions_per_trial=1, 
            project_name="lstm_tuner")
    
        # split the data into train and test
        n_samples = len(X)
        test_size = int(n_samples * (1-train_size))
        n_splits = (n_samples - seq_length) // test_size
        ts_cv = TimeSeriesSplit(n_splits=n_splits,
                            max_train_size= int(len(X) * train_size))
    
        for train_index, test_index in ts_cv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            tuner.search(X_train, y_train, epochs=epoch, validation_data=(X_test, y_test))

        # obtain the best model
        best_model = tuner.get_best_models(num_models=1)[0]

        # fit the best model to all data
        history = best_model.fit(X_train, y_train, epochs=epoch, validation_data=(X_test, y_test))
        history_dict = history.history
        print(history_dict.keys())

        # evaluate the model
        loss = best_model.evaluate(X_test, y_test)
        print(f'Model Validation Loss: {loss}')

        predicted = best_model.predict(X_test)
        
        
        #graph the loss and the accuracy
        # Pérdida de entrenamiento y validación
        # Pérdida de entrenamiento y validación
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        # Número de épocas
        epochs = range(1, len(loss) + 1)

        # Graficar la pérdida de entrenamiento y validación
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, 'bo-', label='Pérdida de Entrenamiento')
        plt.plot(epochs, val_loss, 'ro-', label='Pérdida de Validación')
        plt.title('Pérdida de Entrenamiento y Validación')
        plt.xlabel('Epochs')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()

        # Mostrar algunas predicciones
        plt.plot(y_test, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.legend()
        plt.show()

        

if __name__ == "__main__":
    lstm_model( 
        root_dir="yahoo",
        seq_length=5,
        train_size=0.8,
        epoch=30,)

