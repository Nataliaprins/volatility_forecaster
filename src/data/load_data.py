"""Loads the the corresponding CSV file for the specified stock in processed/ directory.

"""

import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# import the ROOT_DIR_PROJECT from constants.py
from constants import ROOT_DIR_PROJECT


def load_data(stock_name: str, root_dir: str) -> pd.DataFrame:
    """Loads the the corresponding CSV file for the specified stock in processed/ directory.

    Args:
        stock_name (str): Stock name.
        root_dir (str): Root directory.

    Returns:
        pd.DataFrame: Dataframe with the loaded data.

    """

    # Read the file
    df = pd.read_csv(
        os.path.join(ROOT_DIR_PROJECT ,root_dir, "processed","prices", "data_" + f"{stock_name}.csv"),
        parse_dates=True,
        index_col=0,
    )

    print(f"--INFO-- load_data for {stock_name} from {root_dir}.")
    return df

        
if __name__ == "__main__":
    load_data(
        stock_name="googl",
        root_dir="yahoo",
    )