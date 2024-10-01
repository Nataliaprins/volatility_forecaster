import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from volatility_forecaster.constants import project_name
from volatility_forecaster.preprocessing.generate_lagged_data import (
    generate_lagged_data,
)
from volatility_forecaster.pull_data.load_data import load_data
from volatility_forecaster.train_test_split.train_test_data import train_test_data


def ts_train_test_split(
    root_dir, column_name, prod_size, train_size, stock_name, lags, n_splits
):
    train_test_df = train_test_data(
        stock_name,
        column_name,
        prod_size,
        lags,
    )

    x = train_test_df.drop(columns=["log_yield"])  # features
    y = train_test_df["log_yield"]  # target

    # split the data with TimeSeriesSplit

    ts_cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=int(len(x) * train_size))

    for train_index, test_index in ts_cv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"--INFO-- ts_train_test_split for {stock_name} from {root_dir}.")

    return x, y, x_train, x_test, y_train, y_test


if __name__ == "__main__":
    ts_train_test_split(
        root_dir=project_name,
        lags=3,
        n_splits=5,
        train_size=0.8,
        stock_name="googl",
        column_name="log_yield",
        prod_size=0.1,
    )
