"this functions is used to make the experiment with mlflow and the arch model"
import glob
import os

import mlflow
from arch import arch_model
from arch.univariate import GARCH
from mlflow import log_artifact, log_metric, log_param
from sklearn.metrics import mean_squared_error

from volatility_forecaster.arch_model.fit import fit_garch
from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name
from volatility_forecaster.pull_data import load_data
from volatility_forecaster.train_test_split import ts_train_test_split


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
        #TODO: hacer una funciòn que parta solo las Y en train y test
        
        #start the experiment
        with mlflow.start_run() as run:
              
        # train the ARCH model
        model_fit = fit_garch(volatility_type= volatility_type, 
                              stock_name= stock_name, 
                              p= p, 
                              q= q )
    
        #predict the model
        y_pred = model_fit.forecast(horizon=1).variance.iloc[-1]
        y_test = y_test.values
    
        #evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        log_metric("mse", mse)
        log_param("p", p)
        log_param("q", q)
        log_param("volatility_type", volatility_type)
        log_artifact("arch_model_summary.txt")


if __name__ == "__main__":
    make_experiment(volatility_type= "ARCH",
                    root_dir= ROOT_DIR_PROJECT,
                    train_size= 0.8,
                    lags= 1,
                    n_splits= 5,
                    stock_name= "googl",
                    p=1, 
                    q=1)

    
        

