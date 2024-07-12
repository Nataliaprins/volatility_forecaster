"this fuction takes the weights of the portfolio and the "
"current prices of the assets and returns the value of the portfolio"

import glob
import os

import pandas as pd

from src.constants import ROOT_DIR_PROJECT
from src.portfolio.optimize_portfolio import optimize_portfolio


def calculate_portfolio_value(root_dir, pattern):

    #create the folder if it does not exist
    if not os.path.exists(os.path.join(ROOT_DIR_PROJECT, root_dir, "reports", "portfolio")):
        os.makedirs(os.path.join(ROOT_DIR_PROJECT, root_dir, "reports", "portfolio"))

    #obtain weights from optimaze_portfolio function

    weights = optimize_portfolio(root_dir, pattern).values()
    #convert the weights to a list
    weights = list(weights)
    print(weights)

    #obtain prices from the prices.csv file
    #get the path to the prices

    path_to_prices= os.path.join(ROOT_DIR_PROJECT,root_dir,"processed", "prices", "data*.csv")
    price_files = glob.glob(path_to_prices) 
    #interate over the files and read the prices
    df = pd.DataFrame()
    for file in price_files:
        #obtain the file name 
        file_name = file.replace(".csv", "").split("_")[-1]
        df= pd.concat(
            [
                df,
                pd.read_csv(file, index_col=0)[["price"]],
            ],
            axis=1,
        )
        #change the name of the column to the name of the file
        
        df = df.rename(columns={"price": file_name})
    #create a new column in the dataframe with the value of the portfolio multiplied by the weights
    df["portfolio_value"] = df.dot(weights)

    #calculate the percentage change of the portfolio value
    df["portfolio_value_pct_change"] = df["portfolio_value"].pct_change()

    #save the portfolio value to a csv file
    df.to_csv(os.path.join(ROOT_DIR_PROJECT, root_dir, "reports", "portfolio", "portfolio_value.csv"))



if __name__ == '__main__':
    calculate_portfolio_value(root_dir="yahoo", pattern="LinearRegression")



