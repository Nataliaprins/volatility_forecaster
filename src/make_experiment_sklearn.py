"this function is used to make an experiment using mlflow tracking for sk-learn models with the given parameters and choosing the best estimator."

import glob
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from constants import ROOT_DIR_PROJECT
from eval_metrics_mlflow import eval_metrics
from load_test_data_mlflow import load_test_data
from load_train_data_mlflow import load_train_data


def make_experiment(train_size, lags, model_instance, model_params, verbose):
    #choose the best estimator
    estimator = GridSearchCV(model_instance, 
                                 model_params, 
                                 cv=5,
                                 verbose=verbose,
                                 return_train_score=False,
                                 )

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
        
        # obtain x_train, y_train, x_test, y_test
        "TODO: Change for fuction ts_train_test_split"
        
        x_train, y_train = load_train_data(root_dir="yahoo",train_size=189, lags= 5, stock_name= stock_name )
        x_test, y_test = load_test_data(root_dir="yahoo", train_size=189, lags= 5, stock_name= stock_name)

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
        
        # start the experiment
      
        with mlflow.start_run() as run:
            # set the run name            
            run=mlflow.active_run()
            print("Active run_id: {}".format(run.info.run_id))
    
            # train the model
            estimator.fit(x_train, y_train)

            params = estimator.cv_results_['params']
            for i, param_set in enumerate(params):
                param_str = str(param_set)
                mlflow.set_tag("mlflow.runName", f"model: {repr(model_instance)} Run: {i + 1} with params: {param_str}")
                print(f"Parameters for iteration {i + 1}: {param_set}")
            
            # make predictions
            y_pred = estimator.predict(x_test)
            # evaluate the model
            mse, mae, r2 = eval_metrics(y_test, y_pred)
            #get params 
            #param_name = estimator.get_params()
            #print("params: ", param_name)
            #set the run name
            #mlflow.set_tag("mlflow.runName", str(param_name))

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
                    model_instance = DecisionTreeRegressor(), 
                    model_params= {"max_depth":[None, 2]}, 
                    verbose= True)
    print("Experiment made")