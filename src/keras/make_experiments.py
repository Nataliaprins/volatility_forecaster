from src.core._extract_stock_name import _extract_stock_name
from src.core._get_data_files import _get_data_files
from src.keras.make_experiment import make_experiment


def make_experiments(
    project_name,
    model_name,
    model,
    scaler_instance,
    scaler_params,
    seq_length,
    train_size,
    num_max_epochs,
):

    data_files = _get_data_files(project_name=project_name)
    for data_file in data_files:
        stock_name = _extract_stock_name(data_file)

        make_experiment(
            project_name=project_name,
            stock_name=stock_name,
            model_name=model_name,
            model=model,
            scaler_instance=scaler_instance,
            seq_length=seq_length,
            train_size=train_size,
            scaler_params=scaler_params,
            num_max_epochs=num_max_epochs,
        )
