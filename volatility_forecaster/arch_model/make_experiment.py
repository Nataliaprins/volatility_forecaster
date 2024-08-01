"this functions is used to make the experiment with mlflow and the arch model"
import glob
import os

import mlflow
from arch import arch_model
from arch.univariate import GARCH
from mlflow import log_artifact, log_metric, log_param
from sklearn.metrics import mean_squared_error

from volatility_forecaster.arch_model.fit import fit_model
from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name
from volatility_forecaster.core.arch import train_test_split
from volatility_forecaster.mlflow.setting_mlflow import autologging_mlflow
from volatility_forecaster.pull_data import load_data


def make_experiment(volatility_type, root_dir, train_size, lags, n_splits ,stock_name, p=1, q=1 ):
    """
    Fit a GARCH model to the data

    Parameters
    ----------
    data : pd.Series
        The time series of data to fit the GARCH model to
    p : int
        The number of lags of the squared observations to include in the model
    q : int
        The number of lags of the conditional variance to include in the model

    Returns
    -------
    model : arch_model
        The fitted GARCH model
    """
    # obtain the data path
    path_to_data = os.path.join(ROOT_DIR_PROJECT, "data", project_name, "processed", "prices", "*.csv")
    data_files = glob.glob(path_to_data)
    # iterate over the data files
    for data_file in data_files:
        # define the stock name
        stock_name = data_file.split("/")[8].split(".")[0]

        # Load the data
        data = load_data.load_data(stock_name, project_name)
        # obtain Y (retuns)
        returns = data["log_yield"]
        returns = returns.dropna()  
        
        # train test split
        train, test = train_test_split.train_test_split(returns, train_size)

        #Set the experiment
        autologging_mlflow(model_type= volatility_type)

        #set the experiment if it exists
        mlflow.set_experiment(str(stock_name))

        #start the experiment
        with mlflow.start_run() as run:
            #fit the model
            model= fit_model(volatility_type, train, stock_name, p, q)
            #forecast
            forecast = model.forecast(horizon=lags, start= test.index[0], method='analytic')
            y_pred=forecast.variance.tail()
            print(y_pred)
            
            
   


if __name__ == "__main__":
    make_experiment(volatility_type= "ARCH",
                    root_dir= ROOT_DIR_PROJECT,
                    train_size= 0.8,
                    lags= 5,
                    n_splits= 5,
                    stock_name= "googl",
                    p=1, 
                    q=1)

    
        

