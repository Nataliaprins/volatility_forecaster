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

#
# Creates a histogram of the column 'log_yield' using plotly for each
# file in the processed/ directory. The histogram is saved in the
# reports/histogram/ directory. The name of the file is the same as the
# stock name.
#

# Import libraries
import os

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats

from volatility_forecaster.constants import ROOT_DIR_PROJECT


# Import functions
def plot_histogram(
    root_dir: str,
):
    """Plots a histogram of the column 'log_yield' using plotly for each
    file in the processed/ directory. The histogram is saved in the
    reports/histogram/ directory. The name of the file is the same as the
    stock name.

    Args:
        root_dir (str): Root directory.

    """

    # Create the directory if it does not exist
    if not os.path.exists(
        os.path.join(ROOT_DIR_PROJECT, root_dir, "reports/histogram/")
    ):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, root_dir, "reports/histogram/"))

    # Get the list of files
    files = os.listdir(os.path.join(ROOT_DIR_PROJECT, root_dir, "processed", "prices"))

    # Iterate over the files
    for file in files:
        # Skip the .DS_Store file
        if file == ".DS_Store":
            continue
        # Load the data
        df = pd.read_csv(
            os.path.join(ROOT_DIR_PROJECT, root_dir, "processed", "prices", file)
        )
        # drop na values
        df = df.dropna()

        # Create the histogram
        fig = px.histogram(
            df,
            x="log_yield",
            title=f"Histogram of {file[:-4]}",
            labels={"log_yield": "Log yield"},
        )
        # fit a normal distribution to the log_yield
        mu, std = stats.norm.fit(np.array(df["log_yield"]))
        # Create a sequence of numbers from min to max (x-axis)
        x_data = np.linspace(min(df["log_yield"]), max(df["log_yield"]), 100)
        # Create the normal distribution for the range
        normal_dist = stats.norm.pdf(x_data, mu, std)

        # Add the normal distribution to the histogram
        fig.add_scatter(
            x=x_data, y=normal_dist, mode="lines", name="Normal distribution"
        )

        # Save the histogram
        file_name = os.path.join(
            ROOT_DIR_PROJECT, root_dir, f"reports/histogram/{file[:-4]}.html"
        )
        fig.write_html(file_name)

        print(f"--MSG-- File saved to {file_name}")


if __name__ == "__main__":
    plot_histogram(
        root_dir="yahoo",
    )
