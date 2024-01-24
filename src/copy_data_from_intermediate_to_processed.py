"""Copy all files from intermediate/ to processed/. 

>>> from .copy_data_from_intermediate_to_processed import copy_data_from_intermediate_to_processed
>>> copy_data_from_intermediate_to_processed("yahoo")
--MSG-- File saved to msft.csv
--MSG-- File saved to googl.csv
--MSG-- File saved to aapl.csv
--MSG-- All files copied from intermediate/ to processed/prices/

"""
import os
import sys

import pandas as pd
# add the path to the root directory of the project to the sys.path list
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from constants import ROOT_DIR_PROJECT

def copy_data_from_intermediate_to_processed(root_dir):


    """Copy all files from intermediate/ to processed/."""

    # Get the list of files in intermediate/
    intermediate_files = os.listdir(os.path.join(ROOT_DIR_PROJECT, root_dir, "intermediate"))

    #
    # Creates the folder 'processed/prices' inside 'root_dir' if not exists
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, root_dir, "processed", "prices")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, root_dir, "processed", "prices"))

    # Process each file in intermediate/
    for intermediate_file in intermediate_files:
        # Read the file
        if intermediate_file.endswith(".csv"):
            df = pd.read_csv(
                os.path.join(ROOT_DIR_PROJECT, root_dir, "intermediate", intermediate_file),
                parse_dates=True,
                index_col=0,
            )

            # Save the file
            df.to_csv(
                os.path.join(
                    ROOT_DIR_PROJECT, root_dir, "processed/prices", intermediate_file
                ),
                index=True,
            )
            print(f"--MSG-- File saved to {intermediate_file}")

    # Print message
    print("--MSG-- All files copied from intermediate/ to processed/prices/")


if __name__ == "__main__":
    copy_data_from_intermediate_to_processed(root_dir="yahoo")
