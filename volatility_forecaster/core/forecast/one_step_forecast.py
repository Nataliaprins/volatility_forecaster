from volatility_forecaster.arch_model.forecast_one_step_arch import (
    forecast_one_step_arch,
)
from volatility_forecaster.keras.forecast_one_step_keras import forecast_one_step_keras
from volatility_forecaster.sklearn.forecast_one_step_sklearn import (
    forecast_one_step_sklearn,
)
from volatility_forecaster.statsmodels.forecast_one_step_statsmodels import (
    forecast_one_step_statsmodels,
)


def one_step_forecast(
    model_type,
    project_name,
    stock_name,
    lags,
    logged_model_path,
    column_name,
):
    if model_type == "arch":
        prediction = forecast_one_step_arch(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
            column_name=column_name,
        )
        return prediction
    elif model_type == "keras":
        prediction = forecast_one_step_keras(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
        )
        return prediction
    elif model_type == "sklearn":
        prediction = forecast_one_step_sklearn(
            project_name=project_name,
            stock_name=stock_name,
            logged_model_path=logged_model_path,
            lags=lags,
            column_name=column_name,
        )
        return prediction
    elif model_type == "statsmodels":
        prediction = forecast_one_step_statsmodels(
            stock_name=stock_name,
            project_name=project_name,
            logged_model_path=logged_model_path,
        )
        return prediction
    else:
        raise ValueError("--MSG-- model_type not recognized")
