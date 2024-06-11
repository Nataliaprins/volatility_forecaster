"this function is used to make the experiment with keras"
import glob
import os
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from constants import ROOT_DIR_PROJECT
from load_data import load_data
from ts_train_test_split import ts_train_test_split


def make_experiment_keras(root_dir,
        train_size,
        lags,
        model_instance):
    # create the working directory
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, "yahoo", "models", "mlflow")) 

    #obtain data files
    path_to_data = os.path.join(ROOT_DIR_PROJECT, "yahoo", "processed", "prices")
    data_files = glob.glob(os.path.join(path_to_data, "*.csv"))
    
    # obtain a list for the stock name
    stock_names = [Path(file).stem.split("_")[1] for file in data_files]
    
    # iterate over the stock names
    for stock in stock_names:
        df = load_data(stock_name= stock, root_dir= root_dir)
        df = df.dropna()
        serie = df["log_yield"]
        
        #train test split
        x,y= ts_train_test_split(root_dir= root_dir, lags=lags, n_splits=5,
        train_size=train_size, stock_name=stock)
        print(x.shape)
        print(y.shape)
        

        # Use a model LSTM if model_instance is lstm
        if model_instance == "lstm":
            from models.keras.lstm import lstm_model
            model = lstm_model(lags=lags)
            # Use a model GRU if model_instance is gru  
        elif model_instance == "gru":
            from models.keras.gru import gru_model
            model = gru_model(lags=lags)
        elif model_instance == "cnn":
            from models.keras.cnn import cnn_model
            model = cnn_model(lags=lags)
        elif model_instance == "transformer":
            from models.keras.transformer import transformer_model
            model = transformer_model(lags=lags)
        else: 
            raise ValueError("Model instance not available")
        
          

            

if __name__ == "__main__":
    make_experiment_keras(
        root_dir="yahoo",
        train_size=0.8,
        lags=5,
        model_instance="lstm",
    )



    