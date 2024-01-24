# pylint: disable=line-too-long
"""
Este script descarga los precios de cierre ajustados de las acciones de yahoo finance.

--MSG-- Stock prices saved to data/yahoo/raw/aapl.csv
--MSG-- Stock prices saved to data/yahoo/raw/msft.csv
--MSG-- Stock prices saved to data/yahoo/raw/googl.csv

"""

import os

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import yfinance as yf

# from src import constants
from constants import ROOT_DIR_PROJECT

# import sys


def download_data_from_yahoo(
    stocks_list,
    start_date,
    end_date,
    root_dir,
):
    """Download specified stocks from Yahoo Finance and save them to individual CSV files."""

    # Create the root directory if it does not exist
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, root_dir, "raw")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, root_dir, "raw"))
        print(f"--MSG-- Created directory {ROOT_DIR_PROJECT}/{root_dir}/raw")

    # Process each stock in the stocks list. Stock is the nemotechnic of the stock
    for stock_name in stocks_list:
        #
        data = yf.download(stock_name, start=start_date, end=end_date, progress=False)
        data_frame = pd.DataFrame(
            {"price": data["Adj Close"]},
            index=pd.date_range(start=start_date, end=end_date),
        )
        # print(os.path.abspath(os.getcwd()))
        file_path = os.path.abspath("data/yahoo/raw")
        file_path = os.path.join(file_path, "data_" + stock_name.lower() + ".csv")
        # print (file_path)
        data_frame.to_csv(file_path, index=True)
        print(f"--MSG-- Stock prices saved to {file_path}")


# TODO: Revisar el código de este script posiblemente quitar el data del nombre del archivo.

# Ejecutar como: python3 -m download_data_from_yahoo
if __name__ == "__main__":
    download_data_from_yahoo(
        stocks_list=["AAPL", "MSFT", "GOOGL"],
        start_date="2020-01-01",
        end_date="2020-12-31",
        root_dir="yahoo",
    )
