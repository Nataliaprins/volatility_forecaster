import glob
import os

from volatility_forecaster.constants import ROOT_DIR_PROJECT, project_name


def _get_data_files():

    path_to_data = os.path.join(
        ROOT_DIR_PROJECT, "data", project_name, "processed", "prices", "*.csv"
    )
    data_files = glob.glob(path_to_data)
    return data_files
