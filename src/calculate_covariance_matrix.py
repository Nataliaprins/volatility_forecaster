"this fuction will take the y_predicted column of file in folder full and it will calculate the covariance matrix and the correlation matrix of the portfolio"
import glob
import os

import pandas as pd
from pypfopt import risk_models

from constants import ROOT_DIR_PROJECT


def calculate_covariance_matrix(root_dir, pattern):
    # create folder if not exist
    if not os.path.exists(
        os.path.join(ROOT_DIR_PROJECT, root_dir, "reports/covariance_matrix/")
    ):
        os.makedirs(
            os.path.join(ROOT_DIR_PROJECT, root_dir, "reports/covariance_matrix/")
        )

    # obtain the data
    data_file_name = os.path.join(
        ROOT_DIR_PROJECT,
        root_dir,
        "processed",
        "full",
        f"{pattern}*.csv",
    )
    # get the list of files
    files = glob.glob(data_file_name)

    # create a dataframe with the yt_predicted column of each file
    df = pd.DataFrame()
    for file in files:
        df = pd.concat(
            [
                df,
                pd.read_csv(file, index_col=0)[["yt_predicted"]],
            ],
            axis=1,
        )
        # change the name of the column to the name of the file
        df = df.rename(columns={"yt_predicted": file.split("_")[-2]})
    # drop na values
    df = df.dropna()
    print(df.head())

    # calculate the covariance matrix
    covariance_matrix = df.cov()
    print(covariance_matrix)
    # print(covariance_matrix)

    # Save the covariance matrix
    covariance_matrix.to_csv(
        os.path.join(
            ROOT_DIR_PROJECT,
            root_dir,
            "reports/covariance_matrix",
            f"{pattern}_covariance_matrix.csv",
        )
    )

    print("Covariance matrix saved")


if __name__ == "__main__":
    calculate_covariance_matrix(root_dir="yahoo", pattern="LinearRegression")
