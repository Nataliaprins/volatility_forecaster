"this fuction splits the time series data into training and testing data using timeseries_dataset_from_array from tensorflow."
import numpy as np
import tensorflow as tf
from keras.utils import pad_sequences

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name
from volatility_forecaster.pull_data.load_data import load_data


def ts_split_keras(
    stock_name,
    seq_length,
    train_size,
):
    # load the data
    data_frame = load_data(stock_name=stock_name, root_dir=project_name)
    y = data_frame["log_yield"].dropna()

    # convert dataframe to numpy array
    y = y.to_numpy().reshape(-1, 1)

    # padding sequences
    sequences = pad_sequences(y, maxlen=seq_length, dtype="float32", padding="pre")
    # create the dataset X and Y
    X = sequences[:, :-1]
    Y = sequences[:, -1]
    # split the data into training and testing data
    x_train, x_test = X[: int(len(X) * train_size)], X[int(len(X) * train_size) :]
    y_train, y_test = Y[: int(len(Y) * train_size)], Y[int(len(Y) * train_size) :]

    print(
        f"x_train shape: {x_train.shape}"
        f"x_test shape: {x_test.shape}"
        f"y_train shape: {y_train.shape}"
        f"y_test shape: {y_test.shape}"
    )


if __name__ == "__main__":
    ts_split_keras(stock_name="AAPL", seq_length=7, train_size=0.8)
