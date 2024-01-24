# pylint: disable=invalid-name
""" 
Adds a column with the daily log yields of the adjusted close prices.

# >>> from .compute_log_yield  import compute_log_yield 
# >>> compute_log_yield ("yahoo")
--MSG-- File saved to msft.csv
--MSG-- File saved to googl.csv
--MSG-- File saved to aapl.csv
--MSG-- All files processed.

"""

import os

# sys.path.append(os.path.join(os.path.dirname(__file__),'..' ))
import numpy as np
import pandas as pd

from constants import ROOT_DIR_PROJECT


def compute_log_yield(
    root_dir,
):
    """Adds a column with the daily log yields of the adjusted close prices."""

    # Get the list of files in processed/
    processed_files = [
        f
        for f in os.listdir(
            os.path.join(ROOT_DIR_PROJECT, root_dir, "processed/prices/")
        )
        if not f.endswith(".DS_Store")
    ]

    # Process each file in processed/
    for processed_file in processed_files:
        # Read the file
        df = pd.read_csv(
            os.path.join(
                ROOT_DIR_PROJECT, root_dir, "processed", "prices", processed_file
            ),
            parse_dates=True,
            index_col=0,
            encoding="utf-8",
        )

        # Compute the log yields
        df["log_yield"] = np.log(df["price"] / df["price"].shift(1))

        # Save the file
        df.to_csv(
            os.path.join(
                ROOT_DIR_PROJECT, root_dir, "processed/prices/", processed_file
            ),
            index=True,
        )
        print(f"--MSG-- File saved to {processed_file}")

    # Print message
    print("--MSG-- All files processed.")


if __name__ == "__main__":
    compute_log_yield(root_dir="yahoo")
