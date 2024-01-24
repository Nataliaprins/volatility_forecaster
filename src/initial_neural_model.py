" this function initializes the neural network model and loads the weights from the saved model with keras"
# import the libraries
import os
from datetime import datetime

import joblib
from keras.layers import Dense
from keras.models import Sequential

from constants import ROOT_DIR_PROJECT


def initial_neural_model(root_dir):
    # create folder if not exists
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, root_dir, "models")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, root_dir, "models"))

    # Extracts yield names
    processed_files = os.listdir(
        os.path.join(ROOT_DIR_PROJECT, root_dir, "processed", "train")
    )
    processed_files = [
        file
        for file in processed_files
        if file.startswith("train_") and file.endswith(".csv")
    ]
    model_names = [name.replace(".csv", "")[6:] for name in processed_files]

    # Creates a model for each yield
    for name in model_names:
        # Obtains a string with the date and the hour of the creation
        # of the model, to add to the model name
        date_time = datetime.now().strftime("%Y%m%d%H%M%S")

        file_name = os.path.join(
            ROOT_DIR_PROJECT,
            root_dir,
            "models",
            "neural_network_" + date_time + "_" + name + ".joblib",
        )
        # inicialize the model
        neural_model = Sequential()
        neural_model.add(Dense(128, input_dim=5, activation="relu"))
        neural_model.add(Dense(64, activation="relu"))
        neural_model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae"])
        joblib.dump(neural_model, file_name)
        print(f"--MSG-- Models saved to {file_name}")
        print(neural_model.summary())


if __name__ == "__main__":
    initial_neural_model(root_dir="yahoo")
