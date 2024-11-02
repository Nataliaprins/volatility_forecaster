# pylint: disable=invalid-name

"""Plots as a time line the yield column of an specific stock using plotly.

>>> from .plot_yield_time_line import plot_yield_time_line
>>> plot_yield_time_line(
...     stock_name="aapl",
...     root_dir="yahoo",
... ).write_html("reports/appl_yield_time_line.html")

"""

import os

import plotly.graph_objects as go

from volatility_forecaster.constants import ROOT_DIR_PROJECT
from volatility_forecaster.core.directories.create_reports_dir import create_reports_dir
from volatility_forecaster.pull_data.load_data import load_data


def plot_yield_time_line(
    stock_name: str,
    project_name: str,
):
    """Plots as a time line the yield column of an specific stock using plotly.

    Args:
        stock_name (str): Stock name.
        root_dir (str): Root directory.

    """
    create_reports_dir(project_name)

    df = load_data(stock_name, project_name).dropna()
    df = df.reset_index()

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df.index,
                y=df["log_yield"],
                mode="lines",
                line=dict(color="rgb(31, 119, 180)"),
                name="log_yield",
            )
        ]
    )

    fig.update_layout(
        title="Yield time line" + f" {stock_name}",
        xaxis_title="Date",
        yaxis_title="Yield",
        font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
    )

    # Save the plot as html file
    fig.write_html(
        os.path.join(
            ROOT_DIR_PROJECT,
            "data",
            project_name,
            "reports",
            "graphs",
            f"{stock_name}_yield_time_line.html",
        )
    )

    return fig.show()


if __name__ == "__main__":
    plot_yield_time_line(
        stock_name="msft",
        project_name="yahoo",
    ).show()
