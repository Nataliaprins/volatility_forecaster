from volatility_forecaster.arch_model.forecast_n_steps_arch import forecast_n_steps_arch
from volatility_forecaster.keras.forecast_n_steps_keras import forecast_n_step_keras


def n_step_forecast(
    model_type, project_name, stock_name, logged_model_path, column_name, n_steps
):
    """
    this function is used for choosing the correct function to forecast the volatility
    """
    if model_type == "arch":
        forecast_n_steps_arch(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
            column_name=column_name,
            n_steps=n_steps,
        )
    elif model_type == "keras":
        forecast_n_step_keras(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
            n_steps=n_steps,
        )
    elif model_type == "sklearn":
        pass
    elif model_type == "statsmodels":
        pass
    else:
        raise ValueError("--MSG-- model_type not recognized")
