from sklearn.model_selection import TimeSeriesSplit
from src.core._save_files import save_files
from src.train_test_split.train_test_data import train_test_data


def ts_train_test_split(
    project_name,
    column_name,
    train_size,
    stock_name,
    lags,
    n_splits,
):
    train_test_df = train_test_data(
        project_name=project_name,
        stock_name=stock_name,
        column_name=column_name,
        train_size=train_size,
        lags=lags,
    )

    x = train_test_df.drop(columns=["log_yield"])  # features
    y = train_test_df["log_yield"]  # target

    # split the data with TimeSeriesSplit

    ts_cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=int(len(x) * train_size))

    for train_index, test_index in ts_cv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    files = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

    for file_name, file in files.items():
        save_files(
            dataframe=file,
            project_name=project_name,
            processed_folder="train_test",
            model_name="sklearn",
            file_name=f"sklearn_{stock_name}_{file_name}.csv",
        )

    # save_files(
    #    dataframe=x_train,
    #    project_name=project_name,
    #    processed_folder="train_test",
    #    model_name="sklearn",
    #    file_name=f"{stock_name}_x_train.csv",
    # )
    # save_files(
    #    dataframe=x_test,
    #    project_name=project_name,
    #    processed_folder="train_test",
    #    model_name="sklearn",
    #    file_name=f"{stock_name}_x_test.csv",
    # )
    # save_files(
    #    dataframe=y_train,
    #    project_name=project_name,
    #    processed_folder="train_test",
    #    model_name="sklearn",
    #    file_name=f"{stock_name}_y_train.csv",
    # )
    # save_files(
    #    dataframe=y_test,
    #    project_name=project_name,
    #    processed_folder="train_test",
    #    model_name="sklearn",
    #    file_name=f"{stock_name}_y_test.csv",
    # )

    print(f"--INFO-- ts_train_test_split for {stock_name} from {project_name}.")

    return x, y, x_train, x_test, y_train, y_test


if __name__ == "__main__":
    ts_train_test_split(
        project_name="yahoo",
        lags=3,
        n_splits=5,
        train_size=0.8,
        stock_name="googl",
        column_name="log_yield",
    )
