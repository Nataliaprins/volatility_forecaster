from src.core._extract_stock_name import _extract_stock_name
from src.core._get_data_files import _get_data_files
from src.core.directories.create_reports_dir import create_reports_dir
from src.graph.acf_plot import acf_plot


def plot_acfs(project_name: str):
    """
    Iterates over files for grpahting the acf
    """
    create_reports_dir(project_name)

    files = _get_data_files(project_name=project_name)

    for file in files:
        stock_name = _extract_stock_name(data_file=file)
        acf_plot(project_name=project_name, stock_name=stock_name)

    return


if __name__ == "__main__":
    plot_acfs(project_name="yahoo")
