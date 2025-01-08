import pandas as pd

from src.pull_data.load_data import load_data


def extract_serie(
    stock_name,
    project_name,
    column_name,
):
    "Extracts a serie from a dataframe."
    data = load_data(stock_name, project_name)
    serie = data[column_name].dropna()
    serie = pd.DataFrame(serie)

    return serie


if __name__ == "__main__":
    extract_serie(
        stock_name="googl",
        project_name="yahoo",
        column_name="log_yield",
    )
