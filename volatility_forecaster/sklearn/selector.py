"this function is used to estimate the parameters of the model with gridsearchcv"
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor


def selector(model_instance, model_params, n_splits, x, train_size):
    """
    this function is used to estimate the parameters of the model with gridsearchcv
    :param model_instance: the model instance
    :param model_params: the model parameters
    :param verbose: the verbosity of the gridsearchcv
    :param n_splits: the number of splits
    :param x: the features
    :param y: the target
    :return: the best estimator
    """
    select = GridSearchCV(
        model_instance,
        model_params,
        cv=TimeSeriesSplit(n_splits=n_splits, max_train_size=int(len(x) * train_size)),
        verbose=True,
        return_train_score=False,
        scoring="max_error",
        refit=True,
    )

    return select


if __name__ == "__main__":
    model_instance = (LinearRegression(),)
    model_params = ({"fit_intercept": [True, False], "n_jobs": [1, 2, 3]},)
    n_splits = (5,)
    train_size = (0.75,)
