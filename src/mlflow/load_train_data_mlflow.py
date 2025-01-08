"this function loads the train data in folder data/yahoo/processed/train"

import glob
import os

import pandas as pd

from src.constants import ROOT_DIR_PROJECT, project_name


def load_train_data(root_dir, train_size, lags, stock_name):
    """Load train data from the data folder."""
    path_to_train_data = os.path.join(
        ROOT_DIR_PROJECT, root_dir, "processed", "train", "train_*.csv"
    )
    train_data_files = glob.glob(path_to_train_data)

    # read the train data
    for data_file in train_data_files:
        if (
            f"_train_{train_size}"
            and f"_lag_{lags}"
            and f"_{stock_name}.csv" in data_file
        ):
            train_data = pd.read_csv(data_file, index_col=0)
            train_data = train_data.dropna()
            x_train = train_data.drop(columns=["yt"])
            y_train = train_data["yt"]
            print(
                f"--MSG-- x_train, y_train for {data_file.split('_')[-1]} with lags {lags} and train_size {train_size}"
            )

            return x_train, y_train


if __name__ == "__main__":
    load_train_data(root_dir=project_name, train_size=189, lags=5, stock_name="aapl")
