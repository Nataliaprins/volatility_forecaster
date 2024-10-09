# pylint: disable=line-too-long

import pandas as pd

from volatility_forecaster.core.paths.generate_stock_filepath import (
    generate_stock_filepath,
)
from volatility_forecaster.pull_data.download_stock_data import download_stock_data


def download_full_data_from_yahoo(
    stocks_list,
    start_date,
    end_date,
    project_name,
):
    """Download specified stocks from Yahoo Finance and save them to individual CSV files."""

    for stock_name in stocks_list:
        stock_data = download_stock_data(stock_name, start_date, end_date)

        prices_df = pd.DataFrame(
            {"price": stock_data["Adj Close"]},
            index=pd.date_range(start=start_date, end=end_date),
        )

        file_path = generate_stock_filepath(project_name, stock_name)

        prices_df.to_csv(file_path, index=True)
        print(f"--MSG-- Stock prices saved to {file_path}")


if __name__ == "__main__":
    download_full_data_from_yahoo(
        stocks_list=["AAPL", "MSFT", "GOOGL"],
        start_date="2018-01-01",
        end_date="2024-05-31",
        project_name="yahoo",
    )
