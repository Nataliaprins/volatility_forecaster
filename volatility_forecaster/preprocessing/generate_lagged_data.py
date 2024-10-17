import pandas as pd


def generate_lagged_data(serie, lags, column_name):
    serie = pd.DataFrame(serie)
    for lag in range(1, lags + 1):
        serie[f"lag_{lag}"] = serie[str(column_name)].shift(lag)
    return serie
