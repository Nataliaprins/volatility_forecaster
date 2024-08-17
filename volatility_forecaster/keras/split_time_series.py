"this fuction splits the time series data into training and testing data using timeseries_dataset_from_array from tensorflow."
import numpy as np
import tensorflow as tf

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name


def split_time_series(
    X,
    Y,
    train_size,
):

    x_train, x_test = X[: int(len(X) * train_size)], X[int(len(X) * train_size) :]
    y_train, y_test = Y[: int(len(Y) * train_size)], Y[int(len(Y) * train_size) :]

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print(
        f"x_train shape: {x_train.shape}"
        f"x_test shape: {x_test.shape}"
        f"y_train shape: {y_train.shape}"
        f"y_test shape: {y_test.shape}"
    )
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    split_time_series(
        X=np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
                [6, 7, 8],
                [7, 8, 9],
                [8, 9, 10],
                [9, 10, 11],
                [10, 11, 12],
            ]
        ),
        Y=np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
        train_size=0.7,
    )
