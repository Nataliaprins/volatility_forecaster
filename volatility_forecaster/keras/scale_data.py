"this function scales data  using diferenct scalers given the scaler name"
import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def scale_data(data, scaling_method, **scaler_params):
    if scaling_method == "min_max":
        scaler_instance = MinMaxScaler(**scaler_params)
    elif scaling_method == "standard":
        scaler_instance = StandardScaler()
    elif scaling_method == "robust":
        scaler_instance = RobustScaler()
    elif scaling_method == "max_abs":
        scaler_instance = MaxAbsScaler()
    elif scaling_method == "quantile":
        scaler_instance = QuantileTransformer()
    elif scaling_method == "power":
        scaler_instance = PowerTransformer()
    else:
        raise ValueError("The scaler name is not valid")
    scaled_data = scaler_instance.fit_transform(data.values.reshape(-1, 1))
    return scaled_data
