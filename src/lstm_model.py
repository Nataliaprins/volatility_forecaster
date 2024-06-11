"this function is used to create the lstm model using keras tunnig for optimal hyperparameters"
import glob
import os
import sys
from pathlib import Path

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import TimeSeriesSplit

from constants import ROOT_DIR_PROJECT
from create_sequences import create_sequences
from load_data import load_data
from ts_train_test_split import ts_train_test_split


def lstm_model(lags,  root_dir, stock_name):
    # obtain the data path 
    path_to_data = os.path.join(ROOT_DIR_PROJECT, "yahoo", "processed", "prices", "*.csv")
    data_files = glob.glob(path_to_data)
    stock_names = [Path(file).stem.split("_")[1] for file in data_files]

    for stock in stock_names:
        df = load_data(stock_name=stock, root_dir=root_dir)
        df = df.dropna()
        serie= df["rolling_std"]

        #create sequences
        X, y = create_sequences(serie, seq_length=lags)
        print(X.shape, y.shape)

        # define the LSTM model using keras and keras tuner
        def build_model(hp):
            model = Sequential()
            model.add(LSTM(units=hp.Int("units", min_value=32, max_value=512, step=32), input_shape=(lags, 1)))
            model.add(Dense(1))
            model.compile(optimizer=Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])), loss="mse")
            return model    

        tuner= RandomSearch(
            build_model, 
            objective="val_loss", 
            max_trials=3, 
            executions_per_trial=1, 
            project_name="lstm_tuner")
    
        # split the data into train and test
        tscv = ts_train_test_split(n_splits=5)

    
     
   
        


        



if __name__ == "__main__":
    lstm_model(
        lags=5, 
        stock_name="googl", 
        root_dir="yahoo")

