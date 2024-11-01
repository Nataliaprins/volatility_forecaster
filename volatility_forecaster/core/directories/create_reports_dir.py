"this function creates the reports folder in the project"
import os

from volatility_forecaster.constants import ROOT_DIR_PROJECT


def create_reports_dir(project_name):
    if not os.path.exists(
        os.path.join(ROOT_DIR_PROJECT, "data", project_name, "reports")
    ):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data", project_name, "reports"))
        print(f"--MSG-- Folder for reports created successfully")
    return


if __name__ == "__main__":
    create_reports_dir(project_name="yahoo")
