import pandas as pd

from volatility_forecaster.preprocessing.extract_serie import extract_serie
from volatility_forecaster.preprocessing.generate_lagged_data import (
    generate_lagged_data,
)


def train_test_data(
    project_name,
    stock_name,
    column_name,
    train_size,
    lags,
):
    serie = extract_serie(
        stock_name, project_name=project_name, column_name=column_name
    )

    lagged_df = generate_lagged_data(serie, lags, column_name=column_name).dropna()

    train_test_data = lagged_df.head(int(len(lagged_df) * (train_size)))

    train_test_data = pd.DataFrame(train_test_data)

    return train_test_data


if __name__ == "__main__":
    train_test_data(
        project_name="yahoo",
        stock_name="googl",
        column_name="log_yield",
        train_size=0.1,
        lags=3,
    )
