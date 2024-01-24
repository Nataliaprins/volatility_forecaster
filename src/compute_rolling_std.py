# pylint: disable=invalid-name
"""
# >>> from .compute_rolling_std import compute_rolling_std
# >>> compute_rolling_std("yahoo")
--MSG-- File saved to msft.csv
--MSG-- File saved to googl.csv
--MSG-- File saved to aapl.csv
--MSG-- A
"""
import os

import pandas as pd

from constants import ROLLING_WINDOW, ROOT_DIR_PROJECT


def compute_rolling_std(root_dir):
    """Adds a column with the daily yields of the adjusted close prices."""

    # Get the list of files in processed/
    processed_files = os.listdir(
        os.path.join(ROOT_DIR_PROJECT, root_dir, "processed/prices/")
    )

    # Process each file in processed/
    for processed_file in processed_files:
        # Read the file
        df = pd.read_csv(
            os.path.join(
                ROOT_DIR_PROJECT, root_dir, "processed/prices", processed_file
            ),
            parse_dates=True,
            index_col=0,
        )

        # Compute the rolling std
        df["rolling_std"] = df["log_yield"].rolling(window=ROLLING_WINDOW).std()

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
    compute_rolling_std(root_dir="yahoo")
