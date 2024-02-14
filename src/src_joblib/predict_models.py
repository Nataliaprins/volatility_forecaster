# pylint: disable=line-too-long
"""Predict models

# >>> import glob
# >>> from .predict_models import predict_models
# >>> predict_models(
# ...     root_dir="yahoo",
# ...     pattern="LinearRegression*",
# ... )
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_msft_train.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_msft_test.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_msft_full.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_googl_train.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_googl_test.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_googl_full.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_aapl_train.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_aapl_test.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_aapl_full.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_aapl.joblib

# >>> predict_models(
# ...     root_dir="yahoo",
# ...     pattern="*",
# ... )
--MSG-- Forecasts generated for file:
        data/yahoo/processed/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_msft_train.csv
        with model:
        data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_msft_test.csv
        with model:
        data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_msft_full.csv
        with model:
        data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_msft_train.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_msft_test.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_msft_full.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_msft.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_googl_train.csv
        with model:
        data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_googl_test.csv
        with model:
        data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_googl_full.csv
        with model:
        data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_aapl_train.csv
        with model:
        data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_aapl_test.csv
        with model:
        data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_aapl_full.csv
        with model:
        data/yahoo/models/DecisionTreeRegressor_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_googl_train.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_googl_test.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_googl_full.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_googl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_aapl_train.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_aapl_test.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_aapl.joblib
--MSG-- Forecasts generated for file:
        data/yahoo/processed/LinearRegression_yield_lag_5_train_189_test_63_data_aapl_full.csv
        with model:
        data/yahoo/models/LinearRegression_yield_lag_5_train_189_test_63_data_aapl.joblib



"""


import glob
import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from constants import ROOT_DIR_PROJECT


def predict_models(
    root_dir,
    pattern,
):
    """Trains a model based on the model type and model."""

    #
    # Obtains the list of available models in the folder.
    path_to_models = os.path.join(ROOT_DIR_PROJECT, root_dir, "models", pattern)
    model_paths = glob.glob(path_to_models)

    for model_path in model_paths:
        #
        # Obtains the model name
        data_file_name = model_path.replace(".joblib", "")
        data_file_name = "_".join(data_file_name.split("_")[-9:])

        # Loads model
        model = joblib.load(model_path)
        isolated_model_name = model_path.split("/")[-1]

        #
        # Obtains the data
        for particle in ["train", "test", "full"]:
            file_path = os.path.join(
                ROOT_DIR_PROJECT,
                root_dir,
                "processed",
                str(particle),
                str(particle) + "_" + data_file_name + ".csv",
            )
            #
            isolated_data_path = "/".join(file_path.split("/")[:-1])
            #
            data = pd.read_csv(file_path, index_col=0)
            model_data = data.dropna()
            train_x = model_data.drop(columns=["yt"])
            if "yt_predicted" in train_x.columns:
                train_x = train_x.drop(columns=["yt_predicted"])

            #
            # Predicts with the model
            yt_predicted = model.predict(train_x)
            data["yt_predicted"] = pd.NA
            data.loc[train_x.index, "yt_predicted"] = yt_predicted

            #
            # Saves the data
            file_path = (
                isolated_data_path
                + "/"
                + isolated_model_name.replace(".joblib", "_" + particle + ".csv")
            )
            data = data.to_csv(file_path, index=True)

            print("--MSG-- Forecasts generated for file:")
            print(f"        {file_path}")
            print("        with model:")
            print(f"        {model_path}")


if __name__ == "__main__":
    predict_models(
        root_dir="yahoo",
        pattern="neural_network*",
    )
