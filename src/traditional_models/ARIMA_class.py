" this is a class for ARIMA WRAPPER"
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from statsmodels.tsa.arima_model import ARIMA


class ARIMA_class:
    def __init__(self, order):
        self.order = order
        self.model = ARIMA

    def fit(self, x_train, y_train):
        self.model = self.model(y_train, order=self.order)
        self.model_fit = self.model.fit(disp=0)

    def predict(self, x_test):
        y_pred = self.model_fit.forecast(steps=len(x_test))[0]
        return y_pred

    def evaluate(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        msle = mean_squared_log_error(y_test, y_pred)
        return {"mse": mse, "mae": mae, "r2": r2, "mape": mape, "msle": msle}