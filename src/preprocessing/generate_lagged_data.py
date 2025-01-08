import pandas as pd

from src.core._save_files import save_files


def generate_lagged_data(
    project_name,
    stock_name,
    serie,
    lags,
    column_name,
):
    serie = pd.DataFrame(serie)
    for lag in range(1, lags + 1):
        serie[f"lag_{lag}"] = serie[str(column_name)].shift(lag)

    save_files(
        dataframe=serie,
        project_name=project_name,
        processed_folder="lags",
        model_name="sklearn",
        file_name=f"lagged_data_{stock_name}_lag_{lags}.csv",
    )

    return serie
