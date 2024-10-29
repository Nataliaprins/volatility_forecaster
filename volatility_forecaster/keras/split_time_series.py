"this fuction splits the time series data into training and testing data using timeseries_dataset_from_array from tensorflow."
import pandas as pd
from sklearn.model_selection import train_test_split

from volatility_forecaster.core._save_files import save_files


def split_time_series(
    project_name,
    stock_name,
    X,
    Y,
    train_size,
):
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, Y, test_size=(1 - train_size), shuffle=False
    )
    files = {
        "xtrain": xtrain,
        "xtest": xtest,
        "ytrain": ytrain,
        "ytest": ytest,
    }

    for name, file in files.items():
        # aplanar los datos
        if file.ndim > 2:
            file = file.reshape(file.shape[0], file.shape[1])

        save_files(
            dataframe=pd.DataFrame(
                file, columns=[f"feature_{i}" for i in range(file.shape[1])]
            ),
            project_name=project_name,
            processed_folder="train_test",
            model_name="keras",
            file_name=f"{stock_name}_{name}.csv",
        )
        print(f"Saved {name} file for {stock_name}")

    return xtrain, xtest, ytrain, ytest
