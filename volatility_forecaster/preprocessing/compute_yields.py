# pylint: disable=invalid-name
"""Adds a column with the daily yields of the adjusted close prices.

# >>> from .compute_yields import compute_yields
# >>> compute_yields("yahoo")
--MSG-- File saved to msft.csv
--MSG-- File saved to googl.csv
--MSG-- File saved to aapl.csv
--MSG-- All files processed.

"""


import pandas as pd

from volatility_forecaster.core._get_data_files import _get_data_files

# Adds a column called "yields" for each file in data/yahoo/processed/, which
# contains the daily yields of the adjusted close prices.


def compute_yields(
    project_name,
):
    """Adds a column with the daily yields of the adjusted close prices."""

    processed_files = _get_data_files(project_name=project_name)

    for processed_file in processed_files:
        yield_df = pd.read_csv(
            processed_file,
            parse_dates=True,
            index_col=0,
        )

        yield_df["yield"] = yield_df["price"].pct_change()

        yield_df.to_csv(
            processed_file,
            index=True,
        )
        print(f"--MSG-- File saved to {processed_file}")

    print("--MSG-- All files processed.")


if __name__ == "__main__":
    compute_yields(project_name="yahoo")
