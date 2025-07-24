import yfinance as yf


def download_stock_data(stock_name, start_date, end_date):
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    return stock_data


if __name__ == "__main__":
    stock_data = download_stock_data(
        stock_name="^GSPC",
        start_date="2018-01-01",
        end_date="2024-05-31",
    )
    print(stock_data.head())
