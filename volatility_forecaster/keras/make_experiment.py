"this function is used to make the experiment with keras"
import glob
import os
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name
from volatility_forecaster.keras.create_sequences import create_sequences
from volatility_forecaster.keras.scale_data import scale_data
from volatility_forecaster.keras.train_test_split import train_test_split
from volatility_forecaster.pull_data.load_data import load_data


def make_experiment(
    model,
    scaler_instance,
    seq_length,
    train_size,
    scaler_params,
):

    # obtain the data path
    path_to_data = os.path.join(
        ROOT_DIR_PROJECT, "data", project_name, "processed", "prices", "*.csv"
    )
    data_files = glob.glob(path_to_data)

    # iterate over the data files
    for data_file in data_files:
        # define the stock name
        # the data_file contains the path like yahoo/processed/prices/AAPL.csv
        stock_name_with_ext = os.path.basename(data_file)
        stock_name, _ = os.path.splitext(stock_name_with_ext)
        # load Data
        df = load_data(stock_name=stock_name, root_dir=ROOT_DIR_PROJECT)
        df = df.dropna()
        serie = df["log_yield"]
        # scale Data
        scaled_data = scale_data(serie, scaler_instance, **scaler_params)
        # create sequences
        X, y = create_sequences(scaled_data, seq_length)
        # split data
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, seq_length=seq_length
        )
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        # Auto logging
        # set experiment
        mlflow.set_experiment(str(stock_name))
        # start experiment
        with mlflow.start_run() as run:
            # onbtain the model
            model = model
            # obtain the best hyperparameters
            # train the model
            # evaluate the model
            # log the results
            pass
