# pylint: disable=invalid-name
"""
# >>> from .compute_rolling_std import compute_rolling_std
# >>> compute_rolling_std("yahoo")
--MSG-- File saved to msft.csv
--MSG-- File saved to googl.csv
--MSG-- File saved to aapl.csv
--MSG-- A
"""
import pandas as pd

from volatility_forecaster.core._get_data_files import _get_data_files


def compute_rolling_std(project_name, rolling_window):
    """Adds a column with the daily yields of the adjusted close prices."""

    processed_files = _get_data_files(project_name=project_name)

    for processed_file in processed_files:

        std_df = pd.read_csv(
            processed_file,
            parse_dates=True,
            index_col=0,
        )

        std_df["rolling_std"] = std_df["log_yield"].rolling(window=rolling_window).std()

        std_df.to_csv(processed_file, index=True)

        print(f"--MSG-- File saved to {processed_file}")

    print("--MSG-- All files processed.")


if __name__ == "__main__":
    compute_rolling_std(project_name="yahoo", rolling_window=5)
