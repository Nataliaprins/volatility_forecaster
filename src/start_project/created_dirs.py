"this file is used to create a new project folder in the database"

import os
import shutil
import sys

from src.constants import ROOT_DIR_PROJECT

project_name = "yahoo"

def create_project(project_name):  
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, "data", project_name)):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data", project_name))
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data", project_name, "raw"))
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data", project_name, "processed"))
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data", project_name, "intermediate"))
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data", project_name, "features"))
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data", project_name, "models"))
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "data", project_name, "reports"))
        print(f"Folders for project {project_name} created successfully")
    else:
        print(f"Project {project_name} already exists")

create_project(project_name)