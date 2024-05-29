"this function calculates the information ratios for each portfolio in the folder portfolio"
import glob
import os

import pandas as pd
import numpy as np

from constants import ROOT_DIR_PROJECT

from optimize_portfolio import optimize_portfolio


def calculate_ratios_portfolio(root_dir, rf, pattern, betha, rm):
    """
    This function calculates the information ratio for each portfolio in the folder

    """
    # get the list of files in the folder

    path_to_portfolios = os.path.join(
        ROOT_DIR_PROJECT, root_dir, "reports", "portfolio", f"*{pattern}.csv"
    )

    portfolio_files = glob.glob(path_to_portfolios)

    # read the portfolios files
    for file in portfolio_files:
        df_portfolio = pd.read_csv(file, index_col=0)

        # extract the stock names from the columns of the dataframe if the column name starts with 'log_yield'
        stock_list = [
            col for col in df_portfolio.columns if col.startswith("log_yield")
        ]
        # extract the stock name from stock_list
        stock_name = [stock.split("_")[2] for stock in stock_list]
        # create a dictionary to store the average return for each stock in the portfolio
        avg_returns = {}
        for stock in stock_name:
            avg_returns[stock] = df_portfolio[f"log_yield_{stock}"].mean()

    # import the weiths of the optimized portfolio
    weights = optimize_portfolio(root_dir, pattern)

    # multiply the weights by the average returns
    rdto_port = 0
    for stock in stock_name:
        rdto_port += weights[stock] * avg_returns[stock]

    # read the covariance matrix
    cov_matrix = pd.read_csv(
        os.path.join(
            ROOT_DIR_PROJECT,
            root_dir,
            "reports",
            "covariance_matrix",
            f"{pattern}_covariance_matrix.csv",
        ),
        index_col=0,
    )
    print(cov_matrix)

    # multiply the weights by the covariance matrix
    weights_values = np.array(list(weights.values()))
    print(weights_values)
    cov_matrix_values = np.array(cov_matrix)
    print(cov_matrix_values)

    # calculate the product of the weights and the covariance matrix
    product = np.dot(weights_values, cov_matrix_values)
    print(product)
    # transpose the weights list
    weights_values_t = weights_values.reshape(-1, 3)
    print(weights_values_t)
    # multiply the transposed weights by the product
    var_port = np.dot(weights_values_t, product)
    print(var_port)

    # calculate the shrpe ratio
    sharpe_ratio = (rdto_port - float(rf)) / np.sqrt(var_port)
    print(sharpe_ratio)

    # calculate the Treynor ratio
    treynor_ratio = (rdto_port - float(rf)) / betha

    # calculate the Jensen's alpha
    jensen_alpha = rdto_port - (float(rf) + betha * (float(rm) - float(rf)))

    # return the information of portfolio and the weights
    return rdto_port, weights, var_port, sharpe_ratio, treynor_ratio, jensen_alpha


if __name__ == "__main__":
    calculate_ratios_portfolio(
        root_dir="yahoo",
        rf="0.0002",
        pattern="LinearRegression",
        betha=0.5,
        rm="0.0005",
    )
