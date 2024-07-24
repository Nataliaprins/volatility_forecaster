# pylint: disable=line-too-long

import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_models(
    y,
    y_pred,
):
    """Trains a model based on the model type and model."""
    # create the metrics dictionary
    metrics = {
        "mse": mean_squared_error(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
    }
    return metrics

  


