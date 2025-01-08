# pylint: disable=invalid-name
"""Copy all files from raw/ to intermediate/ removing NA.

>>> from .copy_data_from_raw_to_intermediate import copy_data_from_raw_to_intermetidate
>>> copy_data_from_raw_to_intermetidate("yahoo")
--MSG-- File saved to msft.csv
--MSG-- File saved to googl.csv
--MSG-- File saved to aapl.csv
--MSG-- All files copied from raw/ to intermediate/ removing NA.

"""
import os

import pandas as pd

from src.constants import ROOT_DIR_PROJECT


def copy_data_from_raw_to_intermetidate(project_name):
    """Copy all files from raw/ to intermediate/ removing NA."""

    # Get the list of files in raw/
    raw_files = os.listdir(os.path.join(ROOT_DIR_PROJECT, "data", project_name, "raw"))

    # Process each file in raw/
    for raw_file in raw_files:
        # Read the file
        df = pd.read_csv(
            os.path.join(ROOT_DIR_PROJECT, "data", project_name, "raw", raw_file),
            parse_dates=True,
            index_col=0,
            encoding="utf-8",
        )

        # Remove NA
        df.dropna(inplace=True)

        # Save the file
        df.to_csv(
            os.path.join(
                ROOT_DIR_PROJECT, "data", project_name, "intermediate", raw_file
            ),
            index=True,
        )
        print(f"--MSG-- File saved to {raw_file}")

    # Print message
    print("--MSG-- All files copied from raw/ to intermediate/ removing NA.")


if __name__ == "__main__":
    copy_data_from_raw_to_intermetidate(project_name="yahoo")
