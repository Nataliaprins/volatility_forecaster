"""Create train and test matrices of lagged data from a dataset.

# >>> from .train_test_split import train_test_split
# >>> train_test_split(
# ...     yield_type="log_yield", 
# ...     max_lag=5, 
# ...     root_dir="yahoo",
# ... )
--MSG-- File saved to data/yahoo/processed/full_yield_lag_5_train_189_test_63_data_msft.csv
--MSG-- File saved to data/yahoo/processed/train_yield_lag_5_train_189_test_63_data_msft.csv
--MSG-- File saved to data/yahoo/processed/test_yield_lag_5_train_189_test_63_data_msft.csv
--MSG-- File saved to data/yahoo/processed/full_yield_lag_5_train_189_test_63_data_googl.csv
--MSG-- File saved to data/yahoo/processed/train_yield_lag_5_train_189_test_63_data_googl.csv
--MSG-- File saved to data/yahoo/processed/test_yield_lag_5_train_189_test_63_data_googl.csv
--MSG-- File saved to data/yahoo/processed/full_yield_lag_5_train_189_test_63_data_aapl.csv
--MSG-- File saved to data/yahoo/processed/train_yield_lag_5_train_189_test_63_data_aapl.csv
--MSG-- File saved to data/yahoo/processed/test_yield_lag_5_train_189_test_63_data_aapl.csv


"""

import os

import pandas as pd 

#TODO cambiar la ruta de ROOT_DIR_PROJECT por ruta relativa
ROOT_DIR_PROJECT =  "/Users/nataliaacevedo/modelo_tesis_volatilidad/data/yahoo"

def train_test_split(
    yield_type,
    max_lag,
    root_dir,
    test_size=None,
    train_size=None,
    random_state=0,
    shuffle=False,
):
    """TODO:"""
    # Get the list of files in processed/
    processed_files = [f for f in os.listdir(
        os.path.join(root_dir, "processed/prices/")) if not f.endswith('.DS_Store')]

    if not os.path.exists(os.path.join("data", root_dir, "processed/train")):
        os.makedirs(os.path.join("data", root_dir, "processed/train"))

    if not os.path.exists(os.path.join("data", root_dir, "processed/test")):
        os.makedirs(os.path.join("data", root_dir, "processed/test"))

    if not os.path.exists(os.path.join("data", root_dir, "processed/full")):
        os.makedirs(os.path.join("data", root_dir, "processed/full"))

    processed_files = [
        file
        for file in processed_files
        if not file.startswith("train_")
        and not file.startswith("test_")
        and not file.startswith("full_")
        ]
        
         
    #  Process each file in processed/
    for processed_file in processed_files:
        # Read the file
        df = pd.read_csv(
            os.path.join("data", root_dir, "processed/prices/", processed_file),
            parse_dates=True,
            index_col=0,
        )

        yt = df[yield_type].rename("yt")

        train_data_full, _, config_text = apply_train_test_split(
            yt,
            max_lag,
            test_size=None,
            train_size=1.0,
            random_state=None,
            shuffle=False,
        )

        train_data, test_data, config_text = apply_train_test_split(
            yt,
            max_lag,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        full_file_name = (
            "full_" + yield_type + config_text + "_data_" + processed_file[:-4] + ".csv"
        )
        train_file_name = (
            "train_"
            + yield_type
            + config_text
            + "_data_"
            + processed_file[:-4]
            + ".csv"
        )
        test_file_name = (
            "test_" + yield_type + config_text + "_data_" + processed_file[:-4] + ".csv"
        )

        # Save the files
        file_name = os.path.join("data", root_dir, "processed", "full", full_file_name)
        train_data_full.to_csv(file_name, index=True)
        print(f"--MSG-- File saved to {file_name}")

        # Save the files
        file_name = os.path.join("data", root_dir, "processed","train", train_file_name)
        train_data.to_csv(file_name, index=True)
        print(f"--MSG-- File saved to {file_name}")

        # Save the files
        file_name = os.path.join("data", root_dir, "processed", "test", test_file_name)
        test_data.to_csv(file_name, index=True)
        print(f"--MSG-- File saved to {file_name}")


def apply_train_test_split(
    yield_data,  # Pandas series
    max_lag,
    test_size=None,
    train_size=None,
    random_state=0,
    shuffle=False,
):
    """TODO:"""
    yield_data = yield_data.copy()

    #
    # Create a column for each lag
    yield_data = yield_data.to_frame()
    for lag in range(1, max_lag + 1):
        yield_data[f"lag_{lag}"] = yield_data["yt"].shift(-lag)

    #
    # Computes the sizes of train and test sets
    if test_size is not None and train_size is not None:
        raise ValueError("Both test_size and train_size cannot be set.")

    if test_size is None and train_size is None:
        test_size = 0.25
        train_size = 0.75

    if isinstance(test_size, float):
        if test_size > 1:
            raise ValueError("test_size must be between 0 and 1.")
        if test_size < 0:
            raise ValueError("test_size must be between 0 and 1.")
        test_size = int(test_size * len(yield_data))

    if isinstance(train_size, float):
        if train_size > 1:
            raise ValueError("train_size must be between 0 and 1.")
        if train_size < 0:
            raise ValueError("train_size must be between 0 and 1.")
        train_size = int(train_size * len(yield_data))

    if isinstance(test_size, int):
        if test_size > len(yield_data):
            raise ValueError("test_size must be less than the length of the data.")
        if test_size < 0:
            raise ValueError("test_size must be greater than 0.")
        train_size = len(yield_data) - test_size
    elif isinstance(train_size, int):
        if train_size > len(yield_data):
            raise ValueError("train_size must be less than the length of the data.")
        if train_size < 0:
            raise ValueError("train_size must be greater than 0.")
        test_size = len(yield_data) - train_size

    #
    # Shuffle the data
    if shuffle:
        yield_data = yield_data.sample(frac=1, random_state=random_state)

    #
    # Split the data
    train_data = yield_data.iloc[:train_size]
    test_data = yield_data.iloc[train_size:]

    text = (
        "_lag_"
        + str(max_lag)
        + ("_train_" + str(train_size) if train_size is not None else "")
        + ("_test_" + str(test_size) if test_size is not None else "")
    )

    return train_data, test_data, text

if __name__ == "__main__":
    train_test_split(
        yield_type="log_yield",
        max_lag=5,
        root_dir= ROOT_DIR_PROJECT, 
    )
