"this function split the data in train and test set with a given percentage"
from src.core._save_files import save_files


def train_test_split(data, project_name, stock_name, ratio):
    n = len(data)
    train_size = int(n * ratio)
    train = data[:train_size]
    test = data[train_size:]

    save_files(
        dataframe=train,
        project_name=project_name,
        processed_folder="train_test",
        model_name="statsmodels",
        file_name=f"statsmodels_{stock_name}_train.csv",
    )

    save_files(
        dataframe=test,
        project_name=project_name,
        processed_folder="train_test",
        model_name="statsmodels",
        file_name=f"statsmodels_{stock_name}_test.csv",
    )
    return train, test


if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ratio = 0.8
    train, test = train_test_split(data, ratio)
    print(train)
    print(test)
