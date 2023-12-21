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
import sys

import pandas as pd

# TODO cambiar la ruta de ROOT_DIR_PROJECT por ruta relativa
# ROOT_DIR_PROJECT = "/Users/nataliaacevedo/modelo_tesis_volatilidad/data/yahoo"

# obtain the root path of the project
root_path = os.path.abspath(
    os.path.join(os.getcwd(), os.pardir, "modelo_202312", "data")
)


def copy_data_from_raw_to_intermetidate(root_dir):
    """Copy all files from raw/ to intermediate/ removing NA."""

    # Create the intermediate/ directory if not exists
    if not os.path.exists(os.path.join(root_path, root_dir, "intermediate")):
        os.makedirs(os.path.join(root_path, root_dir, "intermediate"))

    # Get the list of files in raw/
    raw_files = os.listdir(os.path.join(root_path, root_dir, "raw"))

    # Process each file in raw/
    for raw_file in raw_files:
        # Read the file
        df = pd.read_csv(
            os.path.join(root_path, root_dir, "raw", raw_file),
            parse_dates=True,
            index_col=0,
            encoding="utf-8",
        )

        # Remove NA
        df.dropna(inplace=True)

        # Save the file
        df.to_csv(
            os.path.join(root_path, root_dir, "intermediate", raw_file),
            index=True,
        )
        print(f"--MSG-- File saved to {raw_file}")

    # Print message
    print("--MSG-- All files copied from raw/ to intermediate/ removing NA.")


if __name__ == "__main__":
    copy_data_from_raw_to_intermetidate(root_dir="yahoo")
