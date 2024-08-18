"this fuction splits the time series data into training and testing data using timeseries_dataset_from_array from tensorflow."

from sklearn.model_selection import train_test_split


def split_time_series(
    X,
    Y,
    train_size,
):
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, Y, test_size=(1 - train_size), shuffle=False
    )

    return xtrain, xtest, ytrain, ytest
