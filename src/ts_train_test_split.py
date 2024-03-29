import glob
import os

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from constants import ROOT_DIR_PROJECT
from load_data import load_data


def ts_train_test_split(root_dir, train_size, stock_name, lags, n_splits):
    # load the data 
    df = load_data(stock_name, root_dir)
    df.dropna(inplace=True)
    # extract the "log_yield" column an rename it to "yt"
    df = df["log_yield"].rename("yt")
    # create a dataframe with the yt column
    lagged_df = pd.DataFrame(df)  
   
    #define the lags
   
   
    for lag in range(1, lags + 1):
        lagged_df[f"lag_{lag}"] = lagged_df["yt"].shift(lag)

    #drop the NaN values
    lagged_df.dropna(inplace=True)
    
    #define X and Y variables
    x = lagged_df.drop(columns=["yt"]) # features
    y = lagged_df["yt"] # target

    #split the data with TimeSeriesSplit
    ts_cv = TimeSeriesSplit(n_splits=n_splits,
                            max_train_size= int(len(x) * train_size))
    for train_index, test_index in ts_cv.split(x):
       x_train, x_test = x.iloc[train_index], x.iloc[test_index]
       y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        

    return  x, y

if __name__ == "__main__":
    ts_train_test_split(
        root_dir= "yahoo",
        lags=3,  
        n_splits=5,
        train_size=0.75, 
        stock_name="googl")
    


      






