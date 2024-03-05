"this function is used to make an experiment with the given parameters using mlflow tracking"

import glob
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression

from constants import ROOT_DIR_PROJECT
from eval_metrics_mlflow import eval_metrics
from load_test_data_mlflow import load_test_data
from load_train_data_mlflow import load_train_data

# path to the data
path_to_train_data = os.path.join(ROOT_DIR_PROJECT, "yahoo", "processed", "train", "train_*.csv")
data_train_files = glob.glob(path_to_train_data)
stock_name = "msft"

x_train, y_train = load_train_data(root_dir="yahoo",train_size=189, lags= 5, stock_name= stock_name )
x_test, y_test = load_test_data(root_dir="yahoo", train_size=189, lags= 5, stock_name= stock_name)

#TODO: hacer el for para iterar entre acciones y agregar el gridsearch en model params para guardar los mejores parámetros.


def make_experiment(train_size, lags, model_instance, model_params, verbose):
    """this function is used to make an experiment with the given parameters using mlflow tracking"""
    #create the working directory
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow" ))

    #setting tracking directory
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
    
    #set the experiment
    mlflow.set_experiment(str(stock_name))
    # start the experiment

    with mlflow.start_run() as run:
        run=mlflow.active_run()
        print("Active run_id: {}".format(run.info.run_id))
   
        # train the model
        model_instance.fit(x_train, y_train)
        # make predictions
        y_pred = model_instance.predict(x_test)
        # evaluate the model
        mse, mae, r2 = eval_metrics(y_test, y_pred)
        # log the parameters
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("lags", lags)
        mlflow.log_param("model_instance", model_instance)
        mlflow.log_param("model_params", model_params)
        # log the metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        # log the model
        mlflow.sklearn.log_model(model_instance, "model")
        
    
if __name__ == "__main__":
    make_experiment(train_size=189,
                    lags= 5, 
                    model_instance= LinearRegression(), 
                    model_params= {"sample_weight": "samples"}, 
                    verbose= True)
    print("Experiment made")