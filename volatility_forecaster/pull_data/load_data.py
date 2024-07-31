"""Loads the the corresponding CSV file for the specified stock in processed/ directory.

"""

import os

import pandas as pd

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name


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
        os.path.join(
            ROOT_DIR_PROJECT,
            "data",
            root_dir,
            "processed",
            "prices",
            f"{stock_name}.csv",
        ),
        parse_dates=True,
        index_col=0,
    )

    print(f"--INFO-- load_data for {stock_name} from {root_dir}.")
    return df


if __name__ == "__main__":
    load_data(
        stock_name="googl",
        root_dir= project_name,
    )
