# pylint: disable=line-too-long
"""Train models

# >>> import glob
# >>> from .train_models import train_models
# >>> # Run models starting with LinearRegression
# >>> train_models(
...     root_dir="yahoo",
...     pattern="LinearRegression*",
... )
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_aapl.joblib

# >>> # Run only one model
# >>> train_models(
...     root_dir="yahoo",
...     pattern="LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib",
... )
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib


# >>> # Run models for msft
# >>> train_models(
# ...     root_dir="yahoo",
# ...     pattern="*msft*",
# ... )
--MSG-- Models saved to data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib

# >>> train_models(
# ...     root_dir="yahoo",
# ...     pattern="*",
# ... )
--MSG-- Models saved to data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Models saved to data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Models saved to data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Models saved to data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_aapl.joblib


"""


import glob
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..' ))

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression 
from constants import ROOT_DIR_PROJECT 


def train_models(
    root_dir,
    pattern,
):
    """Trains a model based on the model type and model."""

    #
    # Obtains the list of available models in the folder.
    path_to_models = os.path.join(ROOT_DIR_PROJECT,root_dir, "models", pattern)
    model_paths = glob.glob(path_to_models)

    for model_path in model_paths:
        #
        # Obtains the model name
        data_file_name = model_path.replace(".joblib", "")
        data_file_name = "_".join(data_file_name.split("_")[3:])

        #
        # Obtains the data
        data_file_name = os.path.join(ROOT_DIR_PROJECT,
            root_dir, "processed","train", "train_"+ data_file_name + ".csv"
        )
        data = pd.read_csv(data_file_name, index_col=0)
        data = data.dropna()
        train_x = data.drop(columns=["yt"])
        train_y = data["yt"]

        #
        # Loads model
        model = joblib.load(model_path)
      

        # Trains model
        model.fit(train_x, train_y)

        #
        # Saves model
        joblib.dump(model, model_path)
        print(f"--MSG-- Models saved to {model_path}")

if __name__ == "__main__":
    train_models(
        root_dir= "yahoo",
        pattern="LinearRegression*",
    )



    