"this function creates the reports folder in the project"
import os

from src.constants import ROOT_DIR_PROJECT


def create_reports_dir(project_name):

    list_report_dir = [
        "forecasting",
        "graphs",
    ]
    for i in list_report_dir:
        if not os.path.exists(
            os.path.join(ROOT_DIR_PROJECT, "data", project_name, "reports", i)
        ):
            os.makedirs(
                os.path.join(ROOT_DIR_PROJECT, "data", project_name, "reports", i)
            )
            print("--MSG-- Folder for reports created successfully")

    return


if __name__ == "__main__":
    create_reports_dir(project_name="yahoo")
