"this function is used to make an experiment using mlflow tracking for sk-learn models with the given parameters and choosing the best estimator."

import glob
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from src.constants import ROOT_DIR_PROJECT, project_name
from src.mlflow.setting_mlflow import autologging_mlflow
from src.traditional_models.selector import selector
from src.train_test_split.ts_train_test_split import ts_train_test_split


def make_experiment(model_type, train_size, lags, model_instance, model_params, root_dir, n_splits):

    # obtain the data path
    path_to_data = os.path.join(ROOT_DIR_PROJECT, "data" ,project_name, "processed", "prices", "*.csv")
    data_files= glob.glob(path_to_data)
    # iterate over the data files
    for data_file in data_files:
        # define the stock name
        stock_name = data_file.split("/")[8].split(".")[0]
       
        # obtain X and Y
        x, y, x_train, x_test, y_train, y_test = ts_train_test_split(root_dir= root_dir, train_size=train_size, lags= lags, stock_name= stock_name, n_splits= n_splits)
        
        #choose the best estimator
        estimator = selector (model_instance, model_params, n_splits,x=x, y=y, train_size=train_size)
                                 
        #autologging
        autologging_mlflow(model_type= "sklearn")          
       
        #set the experiment if it exists
        mlflow.set_experiment(str(stock_name))

        # get the run data
        def fetch_logged_data(run_id):
            client = MlflowClient()
            data = client.get_run(run_id).data
            tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
            return data.params, data.metrics, tags, artifacts

        # start the experiment      
        with mlflow.start_run() as run:
      
            # train the model
            estimator.fit(x, y)                      
            run_id = mlflow.active_run().info.run_id 
                                             
            #show data logged in the run
            params, metrics, tags, artifacts = fetch_logged_data(run_id)
            
            for key, value in metrics.items():
                print(f"Key: {key}, Value: {value}")

            params = estimator.cv_results_['params']
            for i, param_set in enumerate(params):
                param_str = str(param_set)
                mlflow.set_tag("mlflow.runName", f"model: {repr(model_instance)} Run with params: {param_str}")
                   
    print(f"--MSG: Experiment finished for {model_type}--")

if __name__ == "__main__":
    make_experiment(model_type= "sklearn",
                    train_size=0.75,
                    lags= 3, 
                    model_instance = DecisionTreeRegressor(), 
                    model_params= {"max_depth": [1, 2, 3, 4, 5]}, 
                    root_dir= project_name,
                    n_splits=5)
    