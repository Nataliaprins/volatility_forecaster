import os
import pickle

import mlflow.pyfunc
from arch import arch_model


class ArchModelWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, vol, p, q, dist):
        self.vol = vol
        self.p = p
        self.q = q
        self.dist = dist
        self.model_fit = None

    def fit(self, returns, fit_params_combinations=None):
        if fit_params_combinations is None:
            fit_params_combinations = {}
        model = arch_model(returns, vol=self.vol, p=self.p, q=self.q, dist=self.dist)
        self.model_fit = model.fit(**fit_params_combinations)
        return self.model_fit

    def predict(self, context, model_input):
        if self.model_fit is None:
            raise ValueError("The model is not trained yet. Call the fit method first.")

        self.model_fit.forecast(horizon=7).variance.iloc[-1]
        return self.model_fit.forecast(horizon=7).variance.iloc[-1]

    def save_model(self, path):

        if path.startswith("file://"):
            path = path.replace("file://", "")

        model_file_path = os.path.join(path, "arch_model.pkl")

        # serializing the model
        with open(model_file_path, "wb") as f:
            pickle.dump(self.model_fit, f)
        return print(f"Model saved in {model_file_path}")

    @classmethod
    def load_model(cls, path):
        # load the model from a ".pkl" file
        with open(os.path.join(path, "arch_model.pkl"), "rb") as f:
            model_fit = pickle.load(f)
        return cls(model_fit)
