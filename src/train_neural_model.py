"this function trains the neural network model saved in models folder"

import glob
import os

import joblib
import pandas as pd

from constants import ROOT_DIR_PROJECT


def train_neural_model(
    root_dir,
):
    """Trains a model based on the model type and model."""

    # obtain the lis of files in the models folder
    path_to_models = os.path.join(
        ROOT_DIR_PROJECT, root_dir, "models", "neural*.joblib"
    )
    model_paths = glob.glob(path_to_models)

    # obtain the model name from the path
    for model_path in model_paths:
        model_name = model_path.replace(".joblib", "")
        model_name = "_".join(model_name.split("_")[4:])

        # load the model
        model = joblib.load(model_path)

        # obtain the data
        data_file_name = os.path.join(
            ROOT_DIR_PROJECT,
            root_dir,
            "processed",
            "train",
            "train_" + model_name + ".csv",
        )

        data = pd.read_csv(data_file_name, index_col=0)
        data = data.dropna()
        train_x = data.drop(columns=["yt"])
        train_y = data["yt"]

        # train the model
        model.fit(train_x, train_y)

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
