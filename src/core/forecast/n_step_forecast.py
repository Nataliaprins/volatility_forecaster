from src.arch_model.forecast_n_steps_arch import forecast_n_steps_arch
from src.keras.forecast_n_steps_keras import forecast_n_step_keras
from src.sklearn.forecast_n_steps_sklearn import forecast_n_steps_sklearn
from src.statsmodels.forecast_n_steps_statsmodels import forecast_n_steps_statsmodels


def n_step_forecast(
    model_type, project_name, stock_name, logged_model_path, column_name, n_steps, lags
):
    """
    this function is used for choosing the correct function to forecast the volatility
    """
    if model_type == "arch":
        prediction = forecast_n_steps_arch(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
            column_name=column_name,
            n_steps=n_steps,
        )
        return prediction

    elif model_type == "keras":
        prediction = forecast_n_step_keras(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
            n_steps=n_steps,
        )
        return prediction
    elif model_type == "sklearn":
        prediction = forecast_n_steps_sklearn(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
            n_steps=n_steps,
            lags=lags,
            column_name=column_name,
        )
        return prediction
    elif model_type == "statsmodels":
        prediction = forecast_n_steps_statsmodels(
            project_name,
            stock_name,
            logged_model_path,
            n_steps,
        )
        return prediction
    else:
        raise ValueError("--MSG-- model_type not recognized")
