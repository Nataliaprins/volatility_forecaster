from volatility_forecaster.core._extract_stock_name import _extract_stock_name
from volatility_forecaster.core._get_data_files import _get_data_files
from volatility_forecaster.sklearn.make_experiment import make_experiment


def make_experiments(
    project_name,
    model_type,
    column_name,
    train_size,
    lags,
    model_instance,
    param_dict,
    n_splits,
):

    # get the data files
    data_files = _get_data_files(project_name=project_name)
    for data_file in data_files:
        stock_name = _extract_stock_name(data_file)

        make_experiment(
            project_name=project_name,
            stock_name=stock_name,
            model_type=model_type,
            column_name=column_name,
            train_size=train_size,
            lags=lags,
            model_instance=model_instance,
            param_dict=param_dict,
            n_splits=n_splits,
        )
