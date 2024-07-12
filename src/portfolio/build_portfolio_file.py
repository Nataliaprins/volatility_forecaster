"this function builts the portfolio based on the weights and return of stock prices and saves it to a csv file"

import glob
import os

import pandas as pd

from src.constants import ROOT_DIR_PROJECT


def build_portfolio_file(root_dir, pattern):
    # get the list of files in the folder
    path_to_files = os.path.join(
        ROOT_DIR_PROJECT, root_dir, "processed", "prices", "*.csv"
    )
    files = glob.glob(path_to_files)
    # extract log_yield and price columns from each file
    df_rdto = pd.DataFrame()
    for file in files:
        df_rdto = pd.concat(
            [
                df_rdto,
                pd.read_csv(file, index_col=0)[
                    [
                        "price",
                        "log_yield",
                    ]
                ],
            ],
            axis=1,
        )
        # name of the file
        column_name = file.split("/")[-1].split("_")[1].split(".")[0]
        # change the name of the columns to the name of the file
        df_rdto = df_rdto.rename(
            columns={
                "price": "price_" + column_name,
                "log_yield": "log_yield_" + column_name,
            }
        )
    df_rdto = df_rdto.dropna()

    # Extract the yt_predicted column from each file and append to df_rdto
    path_to_model_files = os.path.join(
        ROOT_DIR_PROJECT, root_dir, "processed", "full", f"{pattern}*.csv"
    )
    model_files = glob.glob(path_to_model_files)

    for file in model_files:
        df_rdto = pd.concat(
            [
                df_rdto,
                pd.read_csv(file, index_col=0)[["yt_predicted"]],
            ],
            axis=1,
        )
        # name of the file
        column_name = file.split("data_")[1].split(".")[0].split("_")[0]
        # change the name of the columns to the name of the file
        df_rdto = df_rdto.rename(
            columns={
                "yt_predicted": "yt_predicted_" + column_name,
            }
        )
    print(df_rdto.head())

    # Calculate sharpe ratio for each portfolio

    # save the information ratio to a csv file
    file_name = "portfolio_for_" + pattern + ".csv"
    print(file_name)
    df_rdto.to_csv(
        os.path.join(ROOT_DIR_PROJECT, root_dir, "reports", "portfolio", file_name)
    )

    # print info message
    print(
        f"Portfolio file has been saved to {os.path.join(ROOT_DIR_PROJECT, root_dir, 'reports', 'portfolio', file_name)}"
    )


if __name__ == "__main__":
    build_portfolio_file(root_dir="yahoo", pattern="LinearRegression")
