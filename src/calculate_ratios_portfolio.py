"this function calculates the information ratios for each portfolio in the folder portfolio"
import glob
import os

import pandas as pd

from constants import ROOT_DIR_PROJECT

from optimize_portfolio import optimize_portfolio


def calculate_ratios_portfolio(root_dir, rf, pattern):
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

    # return the information of portfolio and the weights
    return rdto_port, weights


if __name__ == "__main__":
    calculate_ratios_portfolio(root_dir="yahoo", rf="0.02", pattern="LinearRegression")
