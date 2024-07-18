"this function is used to estimate the parameters of the model with gridsearchcv"
import os

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def estimator(model_instance, model_params, verbose, n_splits, x, y):
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
    def estimator(model_instance, model_params, n_splits, x):
        estimator = GridSearchCV(model_instance, 
                             model_params, 
                             cv= TimeSeriesSplit(n_splits= n_splits,max_train_size= int(len(x) * train_size)),
                             verbose=verbose,
                             return_train_score=False,
                             scoring= "max_error", 
                             refit=True,
                             )
    estimator.fit(x, y)
    return estimator.best_estimator_