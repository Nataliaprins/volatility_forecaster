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

import numpy as np
import pandas as pd

from volatility_forecaster.core._get_data_files import _get_data_files


def compute_log_yield(
    project_name,
):
    """Adds a column with the daily log yields of the adjusted close prices."""

    processed_files = _get_data_files(project_name=project_name)

    for processed_file in processed_files:

        log_yield_df = pd.read_csv(
            processed_file,
            parse_dates=True,
            index_col=0,
            encoding="utf-8",
        )

        log_yield_df["log_yield"] = np.log(
            log_yield_df["price"] / log_yield_df["price"].shift(1)
        )

        log_yield_df.to_csv(processed_file, index=True, encoding="utf-8")

        print(f"--MSG-- File saved to {processed_file}")

    return print("--MSG-- All files processed.")


if __name__ == "__main__":
    compute_log_yield(project_name="yahoo")
