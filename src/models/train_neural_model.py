"this function trains the neural network model saved in models folder"

import glob
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from constants import ROOT_DIR_PROJECT


def train_neural_model(
    root_dir,
):
    """Trains a model based on the model type and model."""

    # obtain the lis of files in the models folder
    path_to_models = os.path.join(ROOT_DIR_PROJECT, root_dir, "models", "*.joblib")
    model_paths = glob.glob(path_to_models)

    # obtain the model name from the path
    for model_path in model_paths:
        data_file_name = model_path.replace(".joblib", "")
        data_file_name = "_".join(data_file_name.split("_")[3:])

        # load the model
        model = joblib.load(model_path)

        # obtain the data
        path_to_data = os.path.join(
            ROOT_DIR_PROJECT, root_dir, "data", "train_test", "*"
        )
        data_paths = glob.glob(path_to_data)

        for data_path in data_paths:
            data_name = os.path.basename(data_path)
            data_name = data_name.split(".")[0]
            data_name = data_name.split("_")[0]

            # check if the model name and data name are the same
            if model_name == data_name:
                # load the data
                data = pd.read_csv(data_path)
                # split the data into X and y
                X = data.drop(["Date", "Adj Close"], axis=1)
                y = data["Adj Close"]

                # train the model
                model.fit(X, y)

                # save the model
                path_to_save = os.path.join(
                    ROOT_DIR_PROJECT, root_dir, "models", f"{model_name}.joblib"
                )
                joblib.dump(model, path_to_save)
                print(f"--MSG-- Models saved to {path_to_save}")


if __name__ == "__main__":
    train_neural_model(
        root_dir="yahoo",
    )
