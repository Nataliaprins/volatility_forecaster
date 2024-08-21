import os

from volatility_forecaster.constants import ROOT_DIR_PROJECT
from volatility_forecaster.pull_data.load_data import load_data


def get_stock_series(
    stock_name: str,
):

    # load Data
    df = load_data(stock_name=stock_name, root_dir=ROOT_DIR_PROJECT)
    df = df.dropna()
    serie = df["log_yield"]
    return serie
