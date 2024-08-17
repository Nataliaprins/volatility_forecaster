"this fuction takes data in each stock and creates sequences of data for the lstm model"

import glob
import os

import numpy as np

from volatility_forecaster.pull_data.load_data import load_data


def create_sequences(data, seq_length):

    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys


if __name__ == "__main__":
    create_sequences(
        data=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]), seq_length=3
    )
