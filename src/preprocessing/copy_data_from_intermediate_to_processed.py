"""Copy all files from intermediate/ to processed/. 

>>> from .copy_data_from_intermediate_to_processed import copy_data_from_intermediate_to_processed
>>> copy_data_from_intermediate_to_processed("yahoo")
--MSG-- File saved to msft.csv
--MSG-- File saved to googl.csv
--MSG-- File saved to aapl.csv
--MSG-- All files copied from intermediate/ to processed/prices/

"""
import os

import pandas as pd

from src.constants import ROOT_DIR_PROJECT, project_name


def copy_data_from_intermediate_to_processed():
    """Copy all files from intermediate/ to processed/."""

    # Get the list of files in intermediate/
    intermediate_files = os.listdir( 
        os.path.join(ROOT_DIR_PROJECT, "data" ,project_name, "intermediate")
    )

    #
    # Creates the folder 'processed/prices' inside 'root_dir' if not exists
    if not os.path.exists(
        os.path.join(ROOT_DIR_PROJECT, "data" ,project_name, "processed", "prices")
    ):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data" ,project_name, "processed", "prices"))

    # Process each file in intermediate/
    for intermediate_file in intermediate_files:
        # Read the file
        if intermediate_file.endswith(".csv"):
            df = pd.read_csv(
                os.path.join(
                    ROOT_DIR_PROJECT, "data" ,project_name, "intermediate", intermediate_file
                ),
                parse_dates=True,
                index_col=0,
            )

            # Save the file
            df.to_csv(
                os.path.join(
                    ROOT_DIR_PROJECT, "data" ,project_name, "processed", "prices", intermediate_file
                ),
                index=True,
            )
            print(f"--MSG-- File saved to {intermediate_file}")

    # Print message
    print("--MSG-- All files copied from intermediate/ to processed/prices/")


if __name__ == "__main__":
    copy_data_from_intermediate_to_processed()
