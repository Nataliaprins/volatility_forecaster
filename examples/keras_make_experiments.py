"import the necessary packages"
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from volatility_forecaster.keras.make_experiment import make_experiment

#
seq_length = 7  # sequence length to create inputs and outputs
scaler_instance = "min_max"
scaler_params = {"feature_range": (0, 1)}


# define the LSTM model using keras
length = 5
n_features = 10
out_index = 2

# TODO: crear funcion random search y para simulación
def build_model():
    

make_experiments(
    param_combinations=param_combinations,
    train_size=0.8,
    fit_params=fit_params,
    forecast_params=forecast_params,
)

report_experiment(
    param_combinations=param_combinations,
    train_size=0.8,
    fit_params=fit_params,
    forecast_params=forecast_params,
)
def report_experiment(
    param_combinations: List[Dict[str, Any]],
    train_size: float,
    fit_params: Dict[str, Any],
    forecast_params: Dict[str, Any],
) -> None:
    raise NotImplementedError("This function is not implemenyted e)t."
    

    


# build_model()


# call make_experiment
make_experiment(
    model=build_model(),
    scaler_instance=scaler_instance,
    scaler_params=scaler_params,
    seq_length=seq_length,
    train_size=0.7,
)
