"create a function to plot acf  for a given time series with plotly"
import os

import numpy as np
import plotly.graph_objects as go

from src.constants import ROOT_DIR_PROJECT
from src.core.directories.create_reports_dir import create_reports_dir
from src.pull_data.load_data import load_data
from statsmodels.tsa.stattools import acf


def acf_plot(
    stock_name,
    project_name: str,
    lags=None,
):
    """
    stock: a pandas series
    lags: number of lags to include in the plot

    """
    # Create the directory if it does not exist

    create_reports_dir(project_name)

    # load the data
    df = load_data(stock_name, project_name)
    df = df.reset_index()
    df = df.dropna()

    # convert the dataframe to a time serie
    serie = df["log_yield"]
    # estimate the acf values
    corr_array_acf = acf(serie, alpha=0.05, nlags=lags)
    lower_y_acf = corr_array_acf[1][:, 0] - corr_array_acf[0]
    upper_y_acf = corr_array_acf[1][:, 1] - corr_array_acf[0]

    # create the pacf plot with plotly

    # create the acf plot with plotly
    fig = go.Figure()
    [
        fig.add_scatter(
            x=(x, x), y=(0, corr_array_acf[0][x]), mode="lines", line_color="#3f3f3f"
        )
        for x in range(len(corr_array_acf[0]))
    ]
    fig.add_scatter(
        x=np.arange(len(corr_array_acf[0])),
        y=corr_array_acf[0],
        mode="markers",
        marker_color="#1f77b4",
        marker_size=12,
    )
    fig.add_scatter(
        x=np.arange(len(corr_array_acf[0])),
        y=upper_y_acf,
        mode="lines",
        line_color="rgba(255,255,255,0)",
    )
    fig.add_scatter(
        x=np.arange(len(corr_array_acf[0])),
        y=lower_y_acf,
        mode="lines",
        fillcolor="rgba(32, 146, 230,0.3)",
        fill="tonexty",
        line_color="rgba(255,255,255,0)",
    )
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, 42])
    fig.update_yaxes(zerolinecolor="#000000")
    title = f"Autocorrelation function for {stock_name} stock"
    fig.update_layout(title=title)

    # save the plot
    file_name = os.path.join(
        ROOT_DIR_PROJECT,
        "data",
        project_name,
        "reports/graphs/",
        f"{stock_name}_acf.html",
    )
    fig.write_html(file_name)
    print(f"--MSG-- acf plot for {stock_name} stock was saved in {file_name}")

    # create the pacf plot with plotly

    return fig.show()


if __name__ == "__main__":
    acf_plot(
        stock_name="msft",
        project_name="yahoo",
        lags=40,
    ).show()
