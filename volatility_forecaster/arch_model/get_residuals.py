"this function returns the residuals of the registered model in mlflow"
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from arch import arch_model
from mlflow.pyfunc import load_model

from volatility_forecaster.core.arch._class_ArchModelWrapper import ArchModelWrapper


def get_residuals(model_path, data):
    data = data["log_yield"].dropna()
    # Load the model
    loaded_model = load_model(model_path)
    prediction = loaded_model.predict(pd.DataFrame(data))
    # Get the residuals
    residuals = data - prediction
    print(residuals)


if __name__ == "__main__":
    get_residuals(
        model_path="/Users/nataliaacevedo/volatility_forecaster/data/yahoo/models/mlflow/mlruns/546022347724216931/1e69094578184605b5bf4804fe873c96/artifacts/artifacts",
        data=pd.read_csv(
            os.path.join(
                "/Users/nataliaacevedo/volatility_forecaster/data/yahoo/processed/prices/aapl.csv"
            ),
        ),
    )
