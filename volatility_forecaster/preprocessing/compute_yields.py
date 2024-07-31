# pylint: disable=invalid-name
"""Adds a column with the daily yields of the adjusted close prices.

# >>> from .compute_yields import compute_yields
# >>> compute_yields("yahoo")
--MSG-- File saved to msft.csv
--MSG-- File saved to googl.csv
--MSG-- File saved to aapl.csv
--MSG-- All files processed.

"""

import os

import pandas as pd

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name

# Adds a column called "yields" for each file in data/yahoo/processed/, which
# contains the daily yields of the adjusted close prices.


def compute_yields(
    root_dir,
):
    """Adds a column with the daily yields of the adjusted close prices."""

    # Get the list of files in processed/
    processed_files = os.listdir(
        os.path.join(ROOT_DIR_PROJECT,"data" ,root_dir, "processed/prices/")
    )

    # Process each file in processed/
    for processed_file in processed_files:
        # Read the file
        df = pd.read_csv(
            os.path.join(
                ROOT_DIR_PROJECT, "data" ,root_dir, "processed/prices/", processed_file
            ),
            parse_dates=True,
            index_col=0,
        )

        # Compute the yields
        df["yield"] = df["price"].pct_change()

        # Save the file
        df.to_csv(
            os.path.join(
                ROOT_DIR_PROJECT, "data" ,root_dir, "processed/prices/", processed_file
            ),
            index=True,
        )
        print(f"--MSG-- File saved to {processed_file}")

    # Print message
    print("--MSG-- All files processed.")


if __name__ == "__main__":
    compute_yields(root_dir= project_name)
