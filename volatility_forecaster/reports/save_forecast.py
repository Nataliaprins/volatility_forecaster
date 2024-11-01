"this function saves the reports in the reports folder"
import os

import pandas as pd


def save_forecast(project_name, model_type, stock_name, forecast_n_step_ahead):
    """
    Save the forecast in a csv file
    :param project_name: str
    :param stock_name: str
    :param forecast_one_step_ahead: pd.DataFrame
    :param forecast_n_step_ahead: pd.DataFrame
    :return: None
    """
    forecast_df = pd.DataFrame(forecast_n_step_ahead)

    forecast_df.to_csv(
        os.path.join(
            "data",
            project_name,
            "reports",
            "forecasting",
            f"{model_type}_{stock_name}_n_step_forecast.csv",
        ),
        index=False,
    )
    return print("--MSG-- Forecast saved successfully")


if __name__ == "__main__":
    forecast_n_step_ahead = pd.DataFrame(
        {
            "forecast": [0.01, 0.02, 0.03],
        }
    )
    save_forecast(
        project_name="yahoo",
        model_type="statsmodels",
        stock_name="googl",
        forecast_n_step_ahead=forecast_n_step_ahead,
    )
