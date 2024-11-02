"""Loads the the corresponding CSV file for the specified stock in processed/ directory.

"""

import os

import pandas as pd

from volatility_forecaster.constants import ROOT_DIR_PROJECT


def load_data(stock_name: str, project_name: str) -> pd.DataFrame:
    """Loads the the corresponding CSV file for the specified stock in processed/ directory.

    Args:
        stock_name (str): Stock name.
        root_dir (str): Root directory.

    Returns:
        pd.DataFrame: Dataframe with the loaded data.

    """

    # TODO: revisar que para con los idices al importar el DF

    # Read the file
    prices_df = pd.read_csv(
        os.path.join(
            ROOT_DIR_PROJECT,
            "data",
            project_name,
            "processed",
            "prices",
            f"{stock_name}.csv",
        ),
        parse_dates=True,
        index_col=0,
    )

    print(f"--INFO-- load_data for {stock_name} from {project_name}.")
    return prices_df


if __name__ == "__main__":
    load_data(
        stock_name="googl",
        project_name="yahoo",
    )
