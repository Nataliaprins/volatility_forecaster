"this file is used to create a new project folder in the database"

import os
import shutil
import sys

from volatility_forecaster.constants import ROOT_DIR_PROJECT

# TODO: convert project_name to an atributte of the function


def create_project(project_name):

    dir_name = ["raw", "processed", "intermediate", "features", "models", "reports"]

    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, "data", project_name)):
        for i in dir_name:
            os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data", project_name, i))
        print(f"--MSG-- Folders for project {project_name} created successfully")

    else:
        print(f"--MSG-- Project {project_name} already exists")


create_project("colombia")


if __name__ == "__main__":
    create_project(project_name="colombia")
