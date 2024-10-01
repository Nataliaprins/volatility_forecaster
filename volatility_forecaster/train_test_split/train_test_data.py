import pandas as pd

from volatility_forecaster.preprocessing.extract_serie import extract_serie
from volatility_forecaster.preprocessing.generate_lagged_data import (
    generate_lagged_data,
)


def train_test_data(
    stock_name,
    column_name,
    prod_size,
    lags,
):
    serie = extract_serie(stock_name, column_name)

    lagged_df = generate_lagged_data(lags, serie).dropna()

    train_test_data = lagged_df.head(int(len(lagged_df) * (1 - prod_size)))

    train_test_data = pd.DataFrame(train_test_data)

    return train_test_data


if __name__ == "__main__":
    train_test_prod_split(
        stock_name="googl",
        column_name="log_yield",
        prod_size=0.1,
        lags=3,
    )
