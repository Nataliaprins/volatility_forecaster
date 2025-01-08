"""Plots a histogram of the column 'log_yield' using plotly for each
file in the processed/ directory. The histogram is saved in the
reports/histogram/ directory. The name of the file is the same as the
stock name.

>>> from .histogram import plot_histogram
>>> plot_histogram(
...     root_dir="yahoo",
... )
--MSG-- File saved to data/yahoo/reports/histogram/msft.html
--MSG-- File saved to data/yahoo/reports/histogram/googl.html
--MSG-- File saved to data/yahoo/reports/histogram/aapl.html

"""

import os

import numpy as np
import plotly.express as px
import scipy.stats as stats

from src.constants import ROOT_DIR_PROJECT
from src.core.directories.create_reports_dir import create_reports_dir
from src.pull_data.load_data import load_data


# Import functions
def plot_histogram(
    project_name: str,
    stock_name: str,
):
    """Plots a histogram of the column 'log_yield' using plotly for each
    file in the processed/ directory. The histogram is saved in the
    reports/histogram/ directory. The name of the file is the same as the
    stock name.

    Args:
        root_dir (str): Root directory.

    """
    create_reports_dir(project_name=project_name)

    prices_df = load_data(stock_name=stock_name, project_name=project_name).dropna()

    # Create the histogram
    fig = px.histogram(
        prices_df["log_yield"],
        x="log_yield",
        title=f"Histogram of {stock_name}",
        labels={"log_yield": "Log yield"},
    )
    mu, std = stats.norm.fit(np.array(prices_df["log_yield"]))
    x_data = np.linspace(min(prices_df["log_yield"]), max(prices_df["log_yield"]), 100)
    normal_dist = stats.norm.pdf(x_data, mu, std)
    fig.add_scatter(x=x_data, y=normal_dist, mode="lines", name="Normal distribution")

    # Save the histogram
    file_name = os.path.join(
        ROOT_DIR_PROJECT,
        "data",
        project_name,
        f"reports/graphs/histogram_{stock_name}.html",
    )
    fig.write_html(file_name)
    print(f"--MSG-- File saved to {file_name}")
    return fig.show()


if __name__ == "__main__":
    plot_histogram(
        project_name="yahoo",
        stock_name="googl",
    )
