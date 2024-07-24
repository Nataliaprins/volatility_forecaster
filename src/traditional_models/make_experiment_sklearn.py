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
from src.metrics.evaluate_models import evaluate_models
from src.mlflow.fetch_logged_data import fetch_logged_data
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

        
        
        # start the experiment      
        with mlflow.start_run() as run:
      
            # train the model                   
            estimator.fit(x, y)
            best_model = estimator.best_estimator_
            y_pred = best_model.predict(x_test)

            #evaluate the model
            metrics = evaluate_models(y_test, y_pred)
            #log metrics    
            mlflow.log_metric("mse", metrics["mse"])
            mlflow.log_metric("mae", metrics["mae"])

            best_params = estimator.best_params_

            # get the run data
            params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
            print(params, tags, artifacts)
            
            #log the best model
            mlflow.set_tag("mlflow.runName", f"model: {repr(model_instance)} Run with params: {str(best_params)}")
                   
    print(f"--MSG: Experiment finished for {model_type}--")

if __name__ == "__main__":
    make_experiment(model_type= "sklearn",
                    train_size=0.75,
                    lags= 3, 
                    model_instance = DecisionTreeRegressor(), 
                    model_params= {"max_depth": [1, 2, 3, 4, 5]}, 
                    root_dir= project_name,
                    n_splits=5)
    