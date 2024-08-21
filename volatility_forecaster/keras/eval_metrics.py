from sklearn.metrics import mean_absolute_error, mean_squared_error


def eval_metrics(ytest, ypred):
    mse = mean_squared_error(ytest, ypred)
    mae = mean_absolute_error(ytest, ypred)
    return mse, mae
