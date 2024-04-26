" this function makes runs the statisticts models with mlflow tracking and using Stastsmodels library"

import glob
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error

from constants import ROOT_DIR_PROJECT
from ts_train_test_split import ts_train_test_split


def make_experiment_statsmodels(train_size, lags, model_instance ,model_params, verbose, root_dir, n_splits):    
    
         #create the working directory
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow" ))

    # obtain the data path
    path_to_data = os.path.join(ROOT_DIR_PROJECT, "yahoo", "processed", "prices", "*.csv")
    data_files= glob.glob(path_to_data)

    # setting tracking directory
    artifact_location = Path.cwd().joinpath("data","yahoo","models","mlflow", "mlruns").as_uri()
    mlflow.set_tracking_uri(artifact_location)
    print('Tracking directory:', mlflow.get_tracking_uri())

    # iterate over the data files
    for data_file in data_files:
        # define the stock name
        stock_name = data_file.split("_")[-1].split(".")[0]

    # iterate over the data files
    for data_file in data_files:
        # define the stock name
        stock_name = data_file.split("_")[-1].split(".")[0]
       
        # obtain X and Y
        x, y = ts_train_test_split(root_dir= root_dir, train_size=train_size, lags= lags, stock_name= stock_name, n_splits= n_splits)

    # iterate over the data files
    for data_file in data_files:
        # define the stock name
        stock_name = data_file.split("_")[-1].split(".")[0]
       
        # obtain X and Y
        x, y = ts_train_test_split(root_dir= root_dir, train_size=train_size, lags= lags, stock_name= stock_name, n_splits= n_splits)
    
             #auto logging
        mlflow.statsmodels.autolog(
            log_models=True,
            disable=False,
            exclusive=False,
            disable_for_unsupported_versions=False,
            silent=False,
            registered_model_name=None,            
        )

        mlflow.set_experiment(str(stock_name))

        # get the run data
        def fetch_logged_data(run_id):
            client = MlflowClient()
            data = client.get_run(run_id).data
            tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
            return data.params, data.metrics, tags, artifacts
        
        #start the experiment
        with mlflow.start_run(run_name="statsmodels_"+ stock_name + model_instance) as run:
            #fit the model passed in the argument model_instance
            model = sm.OLS(y, x).fit()


           

            #evaluate the model
            y_pred = model.predict(x)
            mse = mean_squared_error(y, y_pred)
            
            #log the metrics
            mlflow.log_metric("mse", mse)


if __name__ == "__main__":
    make_experiment_statsmodels(
        model_instance= "sm.OLS",
        root_dir= "yahoo",
        train_size= 0.8,
        lags= 5,
        model_params= {},
        verbose= 3,
        n_splits= 5
        )