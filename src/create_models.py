# pylint: disable=line-too-long
"""Creates a generic model to use with any library.

# >>> from sklearn.linear_model import LinearRegression
# >>> from .create_models import create_models
# >>> create_models("yahoo", LinearRegression())
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_googl.joblib

# >>> from sklearn.tree import DecisionTreeRegressor
# >>> create_models("yahoo", DecisionTreeRegressor())
--MSG-- Models saved to data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Models saved to data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Models saved to data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_googl.joblib
"""
import os
from datetime import datetime

import joblib
from sklearn.linear_model import LinearRegression

from constants import ROOT_DIR_PROJECT


def create_models(root_dir, model_instance):
    """Creates a model based on the model type and model."""
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
        #
        # Obtains a string with the date and the hour of the creation
        # of the model, to add to the model name
        date_time = datetime.now().strftime("%Y%m%d%H%M%S")

        file_name = os.path.join(
            ROOT_DIR_PROJECT,
            root_dir,
            "models",
            repr(model_instance).replace("()", "_" + date_time)
            + "_"
            + name
            + ".joblib",
        )
        joblib.dump(model_instance, file_name)
        print(f"--MSG-- Models saved to {file_name}")


if __name__ == "__main__":
    create_models(
        root_dir="yahoo",
        model_instance=LinearRegression(),
    )
