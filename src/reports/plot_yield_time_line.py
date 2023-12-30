# pylint: disable=invalid-name

"""Plots as a time line the yield column of an specific stock using plotly.

>>> from .plot_yield_time_line import plot_yield_time_line
>>> plot_yield_time_line(
...     stock_name="aapl",
...     root_dir="yahoo",
... ).write_html("reports/appl_yield_time_line.html")

"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import plotly.graph_objects as go
import pandas as pd

#import the ROOT_DIR_PROJECT from constants.py
from constants import ROOT_DIR_PROJECT

#import the load_data function from load_data.py
from data.load_data import load_data




def plot_yield_time_line(
    stock_name: str,
    root_dir: str,
):
    """Plots as a time line the yield column of an specific stock using plotly.

    Args:
        stock_name (str): Stock name.
        root_dir (str): Root directory.

    """
    # Create the directory if it does not exist
    os.makedirs(os.path.join(ROOT_DIR_PROJECT,root_dir, "reports/yield_time_line/"), exist_ok=True)

    # df = pd.read_csv(
    #    os.path.join(root_dir, "processed","prices", "data_" + f"{stock_name}.csv"),
    #    parse_dates=True,
    #    index_col=0,
    # )

    # TODO - Natalia: revisar la importación de la función load_data
    df = load_data(stock_name, root_dir)
    df = df.reset_index()
    df = df.sort_values(by="Date")

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df.index,
                y=df["log_yield"],
                mode="lines",
                line=dict(color="rgb(31, 119, 180)"),
            )
        ]
    )

    fig.update_layout(
        title="Yield time line",
        xaxis_title="Date",
        yaxis_title="Yield",
        font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
    )

    return fig


if __name__ == "__main__":
    plot_yield_time_line(
        stock_name="aapl",
        root_dir="yahoo",
    ).show()
