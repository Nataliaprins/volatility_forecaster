"this function takes each model an generates a forecast"
"for a given period an save them in a csv file"
import os
from volatility_forecaster.core.directories import create_reports_dir
from volatility_forecaster.constants import ROOT_DIR_PROJECT 

def report_forecast(project_name):

create_reports_dir(project_name)

#obtener los datos originales 




#guardar los resultados de los y_predichos en un csv 


if __name__ == "__main__":
    create_reports_dir(project_name="yahoo")