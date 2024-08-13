import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def train_test_split(X, y, train_size, seq_length):
    # split the data into train and test
    n_samples = len(X)
    test_size = int(n_samples * (1 - train_size))
    n_splits = (n_samples - seq_length) // test_size

    tscv = TimeSeriesSplit(n_splits=n_splits)

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    return X_train_list, X_test_list, y_train_list, y_test_list
