" this function makes runs the statisticts models with mlflow tracking and using Stastsmodels library"

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow import MlflowClient
from src.constants import ROOT_DIR_PROJECT, project_name
from src.mlflow.setting_mlflow import autologging_mlflow
from src.train_test_split.ts_train_test_split import ts_train_test_split

#TODO: insertar variable model_type

def make_experiment_statsmodels(train_size, lags, model_instance ,model_params, verbose, root_dir, n_splits):    
    
       
    # obtain the data path
    path_to_data = os.path.join(ROOT_DIR_PROJECT,"data",project_name, "processed", "prices", "*.csv")
    data_files= glob.glob(path_to_data)

     # iterate over the data files
    for data_file in data_files:
        # define the stock name
        stock_name = data_file.split("/")[8].split(".")[0]
       
        # obtain X and Y
        x, y, x_train, x_test, y_train, y_test = ts_train_test_split(root_dir= root_dir, train_size=train_size, lags= lags, stock_name= stock_name, n_splits= n_splits)
         
        #auto logging
        autologging_mlflow(model_type= "statsmodels")            
        
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
        root_dir= project_name,
        train_size= 0.8,
        lags= 5,
        model_params= {},
        verbose= 3,
        n_splits= 5
        )