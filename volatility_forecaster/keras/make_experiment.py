"this function is used to make the experiment for each stock using keras"
import glob
import os
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name
from volatility_forecaster.keras.autologging import autologging_mlflow
from volatility_forecaster.keras.create_sequences import create_sequences
from volatility_forecaster.keras.scale_data import scale_data
from volatility_forecaster.keras.split_time_series import split_time_series
from volatility_forecaster.keras.tuning_params import tuning_params
from volatility_forecaster.pull_data.load_data import load_data


def make_experiment(
    model_name,
    model,
    scaler_instance,
    seq_length,
    train_size,
    scaler_params,
    num_max_epochs,
):

    data_files = _get_data_files()

    for data_file in data_files:

        stock_name, serie = get_stock_series(data_file)
        scaled_data = scale_data(serie, scaler_instance, **scaler_params)
        xs, ys = create_sequences(scaled_data, seq_length)
        # TODO: revisar como se esta haciendo el split, y_train no se ajusta a la forma del modelo
        xtrain, xtest, ytrain, ytest = split_time_series(xs, ys, train_size)
        print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

        # reshape the data for LSTM in 3d
        xtrain, xtest = _convert_to_3d(xtrain, xtest)
        print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

        autologging_mlflow()

        mlflow.set_experiment(str(stock_name))

        with mlflow.start_run() as run:

            # obtain the best hyperparameters
            tuner = tuning_params(
                model=model, num_max_epochs=num_max_epochs, model_name=model_name
            )
            print(tuner.search_space_summary())

            tuner.search(xtrain, ytrain, epochs=50, validation_data=(xtest, ytest))
            print(tuner.results_summary())

            best_model = tuner.get_best_models(num_models=1)[0]
            best_model.summary()


def _convert_to_3d(xtrain, xtest):
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)
    return xtrain, xtest

    # train the model
    # evaluate the model
    # log the results


def get_stock_series(data_file):
    # the data_file contains the path like yahoo/processed/prices/AAPL.csv
    stock_name_with_ext = os.path.basename(data_file)
    stock_name, _ = os.path.splitext(stock_name_with_ext)
    # load Data
    df = load_data(stock_name=stock_name, root_dir=ROOT_DIR_PROJECT)
    df = df.dropna()
    serie = df["log_yield"]
    return stock_name, serie


def _get_data_files():

    path_to_data = os.path.join(
        ROOT_DIR_PROJECT, "data", project_name, "processed", "prices", "*.csv"
    )
    data_files = glob.glob(path_to_data)
    return data_files
