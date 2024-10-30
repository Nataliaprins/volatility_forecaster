from volatility_forecaster.arch_model.forecast_one_step_arch import (
    forecast_one_step_arch,
)
from volatility_forecaster.keras.forecast_one_step_keras import forecast_one_step_keras


def one_step_forecast(
    model_type,
    project_name,
    stock_name,
    logged_model_path,
    column_name,
):
    if model_type == "arch":
        forecast_one_step_arch(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
            column_name=column_name,
        )
    elif model_type == "keras":
        forecast_one_step_keras(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
        )
    elif model_type == "sklearn":
        pass
    elif model_type == "statsmodels":
        pass
    else:
        raise ValueError("--MSG-- model_type not recognized")
