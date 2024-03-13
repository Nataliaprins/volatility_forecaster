"this function is used to make the experiment with keras"
import glob
import os
from pathlib import Path

import mlflow
import mlflow.keras

from constants import ROOT_DIR_PROJECT


def make_experiment_keras(
        train_size,
        lags,
        model_instance):
    # create the working directory
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow")) 
    #define network parameters
    n_features = 1
    n_steps = 3
    # define network arquitecture
    neural_model = model_instance
    