"this function is used to make the experiment for each stock using keras"
import os

import mlflow
import mlflow.keras
import numpy as np

from volatility_forecaster.keras._convert_to_3d import _convert_to_3d
from volatility_forecaster.keras._get_experiment_id_by_name import (
    _get_experiment_id_by_name,
)
from volatility_forecaster.keras.autologging import autologging_mlflow
from volatility_forecaster.keras.create_sequences import create_sequences
from volatility_forecaster.keras.eval_metrics import eval_metrics
from volatility_forecaster.keras.scale_data import scale_data
from volatility_forecaster.keras.split_time_series import split_time_series
from volatility_forecaster.keras.tuning_params import tuning_params
from volatility_forecaster.preprocessing.extract_serie import extract_serie


def make_experiment(
    project_name,
    stock_name,
    model_name,
    model,
    scaler_instance,
    seq_length,
    train_size,
    scaler_params,
    num_max_epochs,
):

    serie = extract_serie(
        stock_name=stock_name, project_name=project_name, column_name="log_yield"
    )
    # serie = np.random.rand(2000)
    scaled_data = scale_data(serie, scaler_instance, **scaler_params)
    xs, ys = create_sequences(scaled_data, seq_length)
    xtrain, xtest, ytrain, ytest = split_time_series(
        project_name, stock_name, xs, ys, train_size
    )
    xtrain, xtest = _convert_to_3d(xtrain, xtest)

    autologging_mlflow()

    mlflow.set_experiment(str(stock_name))
    experiment_id = _get_experiment_id_by_name(stock_name)

    experiment = mlflow.get_experiment(experiment_id)
    tuner_directory = os.path.join(
        experiment.artifact_location.replace("file://", ""), "tuner"
    )

    with mlflow.start_run(experiment_id=experiment_id) as run:
        tuner = tuning_params(
            model=model,
            num_max_epochs=num_max_epochs,
            model_name=model_name,
            tuner_directory=tuner_directory,
        )

        tuner.search(xtrain, ytrain, epochs=50, validation_data=(xtest, ytest))

        best_model = tuner.get_best_models(num_models=1)[0]

        ypred = best_model.predict(xtest)

        mse, mae = eval_metrics(ytest, ypred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
