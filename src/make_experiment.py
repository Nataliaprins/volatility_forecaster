"this function is used to make an experiment with the given parameters using mlflow tracking"

import glob
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression

from constants import ROOT_DIR_PROJECT
from load_test_data_mlflow import load_test_data
from load_train_data_mlflow import load_train_data

# path to the data
path_to_train_data = os.path.join(ROOT_DIR_PROJECT, "yahoo", "processed", "train", "train_*.csv")
data_train_files = glob.glob(path_to_train_data)
stock_name = "aapl"

x_train, y_train = load_train_data(root_dir="yahoo",train_size=189, lags= 5, stock_name= stock_name )
x_test, y_test = load_test_data(root_dir="yahoo", train_size=189, lags= 5, stock_name= stock_name)


def make_experiment(train_size, lags, model_instance, model_params, verbose):
    """this function is used to make an experiment with the given parameters using mlflow tracking"""
    #create the working directory
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models","mlflow", "corridas")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models","mlflow", "corridas"))

    tracking_uri = os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow", "corridas")
        
    #setting tracking directory
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
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
        artifact_location= tracking_uri,
        )
    #save the name of the experiment
    mlflow.set_experiment(str(stock_name))
    
    # start the experiment

    with mlflow.start_run() as run:
        run=mlflow.active_run()
        print("Active run_id: {}".format(run.info.run_id))
   
        # train the model
        model_instance.fit(x_train, y_train)
        # make predictions
        y_pred = model_instance.predict(x_test)
        # log the parameters
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("lags", lags)
        mlflow.log_param("model_instance", model_instance)
        mlflow.log_param("model_params", model_params)
        # log the metrics
        mlflow.log_metric("mse", 0.5)
        mlflow.log_metric("mae", 0.5)
        mlflow.log_metric("r2", 0.5)
        # log the model
        mlflow.sklearn.log_model(model_instance, "model")
        
    
if __name__ == "__main__":
    make_experiment(train_size=189,
                    lags= 5, 
                    model_instance= LinearRegression(), 
                    model_params= {"sample_weight": "samples"}, 
                    verbose= True)
    print("Experiment made")