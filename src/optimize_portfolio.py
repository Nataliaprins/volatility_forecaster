"this fuction takes the covariance matrix and optimize a portfolio througth min-variance"
import os
import glob
import pandas as pd
from constants import ROOT_DIR_PROJECT

def optimize_portfolio(root_dir, pattern):
    #obtain the covariance matrix
    covariance_file= os.path.join(ROOT_DIR_PROJECT, 
                                  root_dir, 
                                  "reports", 
                                  "covariance_matrix",
                                  f"{pattern}_covariance_matrix.csv")
    covariance_matrix = pd.read_csv(covariance_file, index_col=0)
    
    #obtain the expected returns
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

    #Optimize the portfolio
    #import the library
    from pypfopt import risk_models
    from pypfopt import expected_returns   
    from pypfopt import EfficientFrontier

    EfficientFrontier(df.mean(), covariance_matrix).max_sharpe()



# /home/natalia/modelo_202312/data/yahoo/reports/covariance_matrix


if __name__ == '__main__':
    optimize_portfolio(root_dir="yahoo",
                       pattern="LinearRegression",)
    