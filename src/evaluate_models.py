# pylint: disable=line-too-long
"""Evaluate models

# >>> import glob
# >>> from .evaluate_models import evaluate_models
# >>> evaluate_models(
# ...     root_dir="yahoo",
# ... )
--INFO-- Metrics saved to data/yahoo/reports/metrics.csv

"""

import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from constants import ROOT_DIR_PROJECT


def evaluate_models(
    root_dir,
):
    """Trains a model based on the model type and model."""

    #create the folder if it does not exist
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, root_dir, "reports", "metrics")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, root_dir, "reports", "metrics"))    

    # specify folders
    folders = ["train", "test", "full"]
    # initialize list of lists
    file_paths = []

    # iterate in the list of folders
    for folder in folders:
        folder_path = os.path.join(ROOT_DIR_PROJECT, root_dir, "processed", str(folder))

        # get the full path with the list of files in the directory
        folder_files = os.listdir(folder_path)
        # choose the files that end in _full.csv, _train.csv, _test.csv
        folder_files = [
            file_name
            for file_name in folder_files
            if file_name.endswith("_full.csv")
            or file_name.endswith("_train.csv")
            or file_name.endswith("_test.csv")
        ]
        # get the full path of the files
        folder_files = [
            os.path.join(folder_path, file_name) for file_name in folder_files
        ]

        # add the file path to list
        file_paths.extend(folder_files)

    metrics = []
    for file_path in file_paths:
        #
        # Obtains the data
        data = pd.read_csv(file_path, index_col=0)

        # Computes the mse and mad using sklearn for the columns 'yt' and 'yt_predicted'.
        data = data[["yt", "yt_predicted"]].dropna()
        yt = data["yt"]
        yt_predicted = data["yt_predicted"]
        metrics.append(
            {
                "file_path": file_path,
                "mse": mean_squared_error(yt, yt_predicted),
                "mad": mean_absolute_error(yt, yt_predicted),
            }
        )

    metrics = pd.DataFrame(metrics).sort_values("file_path")
    path_to_metrics = os.path.join(ROOT_DIR_PROJECT, root_dir, "reports", "metrics/metrics.csv")
    metrics.to_csv(path_to_metrics, index=False)
    print(f"--INFO-- Metrics saved to {path_to_metrics}")


if __name__ == "__main__":
    evaluate_models(
        root_dir="yahoo",
    )
