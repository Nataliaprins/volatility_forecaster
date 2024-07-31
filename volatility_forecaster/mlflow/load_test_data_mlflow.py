"this function is used to load test data from the data folder"

import glob
import os

import pandas as pd

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name


def load_test_data(root_dir, train_size, lags, stock_name):
    """Load test data from the data folder."""
    path_to_test_data = os.path.join(
        ROOT_DIR_PROJECT, root_dir, "processed", "test", "test_*.csv"
    )
    test_data_files = glob.glob(path_to_test_data)

    # read the test data
    for data_file in test_data_files:
        if f"_train_{train_size}" and f"_lag_{lags}" and f"_{stock_name}.csv" in data_file:
            test_data = pd.read_csv(data_file, index_col=0)
            test_data = test_data.dropna()
            x_test = test_data.drop(columns=["yt"])
            y_test = test_data["yt"]
            print(
                f"--MSG-- x_test, y_test for {data_file.split('_')[-1]} with lags {lags} and train_size {train_size}"
            )
            
            return x_test, y_test
        
        
if __name__ == "__main__":
    load_test_data(
        root_dir="project_name",
        train_size=189,
        lags=5,
        stock_name="aapl"
    )
