import pandas as pd


def generate_lagged_data(lags, lagged_df):
    lagged_df = pd.DataFrame(lagged_df)
    for lag in range(1, lags + 1):
        lagged_df[f"lag_{lag}"] = lagged_df["log_yield"].shift(lag)
    return lagged_df
