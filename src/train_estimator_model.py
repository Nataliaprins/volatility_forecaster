"this function is used to train the model using the estimator keras model, and save the model with mlflow tracking"

import glob
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression

from constants import ROOT_DIR_PROJECT
from eval_metrics_mlflow import eval_metrics
from load_test_data_mlflow import load_test_data
from load_train_data_mlflow import load_train_data



def train_estimator_model(
    root_dir,
    train_size,
    lags,
    model_instance,
    verbose
):
    
    #obtain the path for data 
    path_to_train_data= os.path.join(ROOT_DIR_PROJECT,
                                     root_dir,
                                     "processed",
                                     "train",
                                     "train_*.csv")
    data_train_files = glob.glob(path_to_train_data)

    # read the train data
    for data_file in data_train_files:
        x_train, y_train = load_train_data(
        root_dir= root_dir,
        train_size= train_size,
        lags= lags,
        stock_name= data_file.split('_')[-1].replace(".csv", "")
        )
    # read the test data
        x_test, y_test = load_test_data(
        stock_name= data_file.split('_')[-1].replace(".csv", ""),
        root_dir= root_dir,
        train_size= train_size,
        lags= lags
        )

    model_params= {"sample_weight":samples}
    
    #start mlflow run
        print('Tracking directory:', mlflow.get_tracking_uri())

        #mlflow.set_experiment("stock_name")
        #with mlflow.start_run(run_name="linear_regression") as run:
        

        with mlflow.start_run():
            # model.set_params(**params)
            model= (model_instance)           
            model = model.fit(x_train, y_train)      
            mse, mae, r2 = eval_metrics(y_true = y_test, 
                                        y_pred=model.predict(x_test))
            if verbose > 0:
                print(f"--MSG-- mse: {mse} for {data_file.split('_')[-1].replace('.csv', '')}")
                print(f"--MSG-- mae: {mae} for {data_file.split('_')[-1].replace('.csv', '')}")
                print(f"--MSG-- R2: {r2} for {data_file.split('_')[-1].replace('.csv', '')}")
                #save the model
                #tracking the parameters


                #pasar un dic con los parametros e itero en keys y values para grabar
                # for key, value in params.items():
                #     mlflow.log_param(key, value)
                mlflow.log_param("train_size", train_size) #estas no las necesito
                mlflow.log_param("lags", lags) 
                #tracking the metrics
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                #log the model 
                mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    train_estimator_model(
        root_dir="yahoo",
        train_size= "189",
        lags= 5,
        model_instance= LinearRegression(),
        verbose= True
        model_params= {"sample_weight":samples}
    )