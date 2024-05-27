"this function calculates the information ratios for each portfolio in the folder portfolio"
import glob
import os

import pandas as pd

from constants import ROOT_DIR_PROJECT


def calculate_ratios_portfolio(root_dir, rf):
    """
    This function calculates the information ratio for each portfolio in the folder
    
    """
    # get the list of files in the folder
    path_to_portfolios= os.path.join(ROOT_DIR_PROJECT,root_dir,"data","reports","portfolio", "*.csv")
    portfolio_files = glob.glob(path_to_portfolios) 
    
    #Calculate sharpe ratio for each portfolio
    for file in portfolio_files:
        #read the file
        df = pd.read_csv(file, index_col=0)
        #calculate th average return of the portfolio
        avg_return = df["portfolio_value_pct_change"].mean()


        #df["excess_return"] = df["portfolio_value_pct_change"] - float(rf)
        #calculate the standard deviation of the excess return
        std_excess_return = df["excess_return"].std()
        #calculate the information ratio
        df["information_ratio"] = df["excess_return"] / std_excess_return
        #save the information ratio to a csv file
        df.to_csv(ROOT_DIR_PROJECT + root_dir + "/reports/portfolio/" + file)

  
  
    return None

if __name__ == "__main__":
    calculate_ratios_portfolio(root_dir="yahoo", rf="0.02")