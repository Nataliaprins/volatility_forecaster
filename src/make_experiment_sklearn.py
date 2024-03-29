"this function is used to make an experiment using mlflow tracking for sk-learn models with the given parameters and choosing the best estimator."

import glob
import os
from pathlib import Path
from pprint import pprint

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from constants import ROOT_DIR_PROJECT
from eval_metrics_mlflow import eval_metrics
from load_data import load_data
from load_test_data_mlflow import load_test_data
from load_train_data_mlflow import load_train_data
from ts_train_test_split import ts_train_test_split


def make_experiment(train_size, lags, model_instance, model_params, verbose, root_dir, n_splits):


    #create the working directory
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow" ))

    # obtain the data path
    path_to_data = os.path.join(ROOT_DIR_PROJECT, "yahoo", "processed", "prices", "*.csv")
    data_files= glob.glob(path_to_data)

    # iterate over the data files
    for data_file in data_files:
        # define the stock name
        stock_name = data_file.split("_")[-1].split(".")[0]
       
        # obtain X and Y
        x, y = ts_train_test_split(root_dir= root_dir, train_size=train_size, lags= lags, stock_name= stock_name, n_splits= n_splits)
        
        #choose the best estimator
        estimator = GridSearchCV(model_instance, 
                                 model_params, 
                                 cv= TimeSeriesSplit(n_splits= n_splits,max_train_size= int(len(x) * train_size)),
                                 verbose=verbose,
                                 return_train_score=False,
                                 scoring= "max_error", 
                                 refit=True,
                                 )          
           
        # setting tracking directory
        artifact_location = Path.cwd().joinpath("data","yahoo","models","mlflow", "mlruns").as_uri()
        mlflow.set_tracking_uri(artifact_location)
        print('Tracking directory:', mlflow.get_tracking_uri())

                          
        #autologging
        mlflow.sklearn.autolog(
            log_input_examples= False,
            log_model_signatures=True,
            log_models=True,
            disable=False,
            exclusive=False,
            disable_for_unsupported_versions=False,
            silent=False,
            max_tuning_runs=10,
            log_post_training_metrics=True,
            serialization_format= "cloudpickle",
            registered_model_name=None,
            )

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
            # set the run name            
            #run=mlflow.active_run()
            #print("Active run_id: {}".format(run.info.run_id))
    
            # train the model
            estimator.fit(x, y)                      
            run_id = mlflow.active_run().info.run_id 
            parametros = pd.DataFrame(estimator.cv_results_)  
            print(parametros)
            #show data logged in the run
            params, metrics, tags, artifacts = fetch_logged_data(run_id)
            
            for key, value in metrics.items():
                print(f"Key: {key}, Value: {value}")


            #params = estimator.cv_results_['params']
            for i, param_set in enumerate(params):
                param_str = str(param_set)
                mlflow.set_tag("mlflow.runName", f"model: {repr(model_instance)} Run: {i + 1} with params: {param_str}")
                print(f"Parameters for iteration {i + 1}: {param_set}")
            
       
if __name__ == "__main__":
    make_experiment(train_size=0.75,
                    lags= 3, 
                    model_instance = LinearRegression(), 
                    model_params= {"fit_intercept": [True, False], "n_jobs": [1, 2, 3]}, 
                    verbose= True, 
                    root_dir= "yahoo",
                    n_splits=5)
    print("Experiment made")