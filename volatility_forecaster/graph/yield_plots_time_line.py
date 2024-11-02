from volatility_forecaster.core._extract_stock_name import _extract_stock_name
from volatility_forecaster.core._get_data_files import _get_data_files
from volatility_forecaster.core.directories.create_reports_dir import create_reports_dir
from volatility_forecaster.graph.plot_yield_time_line import plot_yield_time_line


def plot_acfs(project_name: str):
    """
    Iterates over files for grpahting the acf
    """
    create_reports_dir(project_name)

    files = _get_data_files(project_name=project_name)

    for file in files:
        stock_name = _extract_stock_name(data_file=file)
        plot_yield_time_line(project_name=project_name, stock_name=stock_name)

    return


if __name__ == "__main__":
    plot_acfs(project_name="yahoo")
