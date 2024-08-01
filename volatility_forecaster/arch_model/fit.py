"this function fits a GARCH model to the data, using the ARCH library"
import os

from arch import arch_model
from arch.univariate import GARCH
from sklearn.metrics import mean_squared_error

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name
from volatility_forecaster.pull_data import load_data


def fit_model(volatility_type, y, stock_name, p=1, q=1 ):
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
    # Load the data
    data = load_data.load_data(stock_name, project_name)
    returns = data["log_yield"]
    returns = returns.dropna()

    # Fit the GARCH model
    model = arch_model (returns, p=p, q=q, mean='Zero', vol= volatility_type)
    model_fit = model.fit(disp='off')
    print(model_fit.summary())
    print(f"fit model {volatility_type} with p={p} and q={q} for {stock_name}")
    return model_fit

if __name__ == "__main__":
    fit_model(  stock_name= "googl",
                p=1, 
                q=1,
                volatility_type= "ARCH")