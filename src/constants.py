import os

# obtain the root directory of the project
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT_DIR_PROJECT = os.path.join(root)  # obtain the root directory of the project
print(ROOT_DIR_PROJECT)
project_name = "yahoo"  # name of the project

ROLLING_WINDOW = 5
