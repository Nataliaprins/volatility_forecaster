import os

from volatility_forecaster.constants import ROOT_DIR_PROJECT


def generate_stock_filepath(project_name, stock_name):
    file_path = os.path.join(
        ROOT_DIR_PROJECT, "data", project_name, "raw", stock_name.lower() + ".csv"
    )
    print(file_path)
    return file_path


if __name__ == "__main__":
    generate_stock_filepath(
        project_name="yahoo",
        stock_name="AAPL",
    )
