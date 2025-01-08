"this functions takes a dataframe and save in a path especified by the user using pandas"

import os

import pandas as pd

from src.constants import ROOT_DIR_PROJECT


def save_files(
    dataframe,
    project_name,
    processed_folder,
    file_name,
    model_name,
):

    path = os.path.join(
        ROOT_DIR_PROJECT,
        "data",
        project_name,
        "processed",
        processed_folder,
        model_name,
    )

    if not os.path.exists(os.path.join(path)):
        os.makedirs(
            os.path.join(
                ROOT_DIR_PROJECT,
                "data",
                project_name,
                "processed",
                processed_folder,
                model_name,
            )
        )

    dataframe.to_csv(os.path.join(path, file_name), index=False)
    return


if __name__ == "__main__":
    save_files(
        dataframe=pd.DataFrame(),
        project_name="yahoo",
        processed_folder="train_test",
        model_name="arch",
        file_name="test.csv",
    )
