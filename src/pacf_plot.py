"create a function to plot pacf for a given time series with plotly"
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import pacf

from constants import ROOT_DIR_PROJECT
from load_data import load_data


def pacf_plot(
    stock_name,
    root_dir: str,
    lags=None,
    type=str,
    both=False,
    alpha=0.05,
    figsize=(10, 5),
):
    """
    stock: a pandas series
    lags: number of lags to include in the plot
    acf: whether to plot acf
    pacf: whether to plot pacf
    both: whether to plot both acf and pacf
    alpha: significance level for confidence intervals
    figsize: size of the figure
    """
    # Create the directory if it does not exist
    if not os.path.exists(
        os.path.join(ROOT_DIR_PROJECT, root_dir, "reports/acf_pacf_plot/")
    ):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, root_dir, "reports/acf_pacf_plot/"))

    # load the data
    df = load_data(stock_name, root_dir)
    df = df.reset_index()
    df = df.dropna()

    # convert the dataframe to a time serie
    serie = df["log_yield"]

    # estimate the pacf values
    corr_array_pacf = pacf(serie, alpha=0.05, nlags=lags)
    lower_y_pacf = corr_array_pacf[1][:, 0] - corr_array_pacf[0]
    upper_y_pacf = corr_array_pacf[1][:, 1] - corr_array_pacf[0]

    # create the pacf plot with plotly

    # create the acf plot with plotly
    fig = go.Figure()
    [
        fig.add_scatter(
            x=(x, x), y=(0, corr_array_pacf[0][x]), mode="lines", line_color="#3f3f3f"
        )
        for x in range(len(corr_array_pacf[0]))
    ]
    fig.add_scatter(
        x=np.arange(len(corr_array_pacf[0])),
        y=corr_array_pacf[0],
        mode="markers",
        marker_color="#1f77b4",
        marker_size=12,
    )
    fig.add_scatter(
        x=np.arange(len(corr_array_pacf[0])),
        y=upper_y_pacf,
        mode="lines",
        line_color="rgba(255,255,255,0)",
    )
    fig.add_scatter(
        x=np.arange(len(corr_array_pacf[0])),
        y=lower_y_pacf,
        mode="lines",
        fillcolor="rgba(32, 146, 230,0.3)",
        fill="tonexty",
        line_color="rgba(255,255,255,0)",
    )
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, 42])
    fig.update_yaxes(zerolinecolor="#000000")
    title = f"Partial Autocorrelation function for {stock_name} stock"
    fig.update_layout(title=title)

    # save the plot
    file_name = os.path.join(
        ROOT_DIR_PROJECT, root_dir, "reports/acf_pacf_plot/", f"{stock_name}_pacf.html"
    )
    fig.write_html(file_name)
    print(f"--MSG-- acf plot for {stock_name} stock was saved in {file_name}")

    # create the pacf plot with plotly

    return fig


if __name__ == "__main__":
    pacf_plot(stock_name="msft", root_dir="yahoo", lags=40, alpha=0.05).show()
