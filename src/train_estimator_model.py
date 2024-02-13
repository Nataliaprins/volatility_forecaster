"this function is used to train the model using the estimator keras model, and save the model with mlflow tracking"

import glob
import os

import mlflow
import pandas as pd

from constants import ROOT_DIR_PROJECT


#obtain data from the data folder
def train_estimator_model(
    root_dir,
    train_size,
    lags
):
    #obtain the train data from the data folder
    path_to_train_data= os.path.join(ROOT_DIR_PROJECT,
                                     root_dir,
                                     "processed",
                                     "train",
                                     "train_*.csv")
    data_train_files = glob.glob(path_to_train_data)

    # read the train data
    for data_file in data_train_files:
        if f"train_{train_size}" and f"_lag_{lags}" in data_file:
            train_data = pd.read_csv(data_file)
            train_data = train_data.dropna()
            x_train= train_data.drop(columns=["yt"]) 
            y_train= train_data["yt"]
            print(f"--MSG-- x_train, y_train for {data_file.split('_')[-1]} with lags {lags} and train_size {train_size}")

            # train the model
            with mlflow.start_run():
                #train the model
                model = model.fit(x_train, y_train)
                #save the model
                model.save(os.path.join(ROOT_DIR_PROJECT, root_dir, "models", "model.h5"))
                print(f"--MSG-- Model saved to {ROOT_DIR_PROJECT}/{root_dir}/models/model.h5")
                #log the model
                mlflow.log_artifact(os.path.join(ROOT_DIR_PROJECT, root_dir, "models", "model.h5"))
                print(f"--MSG-- Model saved to {ROOT_DIR_PROJECT}/{root_dir}/models/model.h5")
                mlflow.end_run()



    

  





    #for data_file in data_files:
    #    #read only the train data in data_file
    #    if "/train/" in data_file:
    #        train_data = pd.read_csv(data_file)
    #        train_data = train_data.dropna()
    #        x_train= train_data.drop(columns=["yt"]) 
    #        y_train= train_data["yt"]
    #        
    #    elif "/test/" in data_file:
    #        test_data = pd.read_csv(data_file)
    #        test_data = test_data.dropna()
    #        x_test= test_data.drop(columns=["yt"])
    #        y_test= test_data["yt"]
    #        #print(f"--MSG-- x_test, y_test for {data_file.split('_')[-1]}")
    #    
            
       
            
           

   
    
    
    ##train the model
    #with mlflow.start_run():
    #    #train the model
    #    model = model.fit(x_train, y_train)
    #    #evaluate the model
    #    score = model.evaluate(x_test, y_test)
    #    #log the model
    #    mlflow.log_param("model", model)
    #    mlflow.log_metric("score", score)
    #    mlflow.log_artifact(os.path.join(ROOT_DIR_PROJECT, root_dir, "processed", "train", "train_data.csv"))
    #    mlflow.log_artifact(os.path.join(ROOT_DIR_PROJECT, root_dir, "processed", "test", "test_data.csv"))
    #    mlflow.log_artifact(os.path.join(ROOT_DIR_PROJECT, root_dir, "models", "model.h5"))
    #    print(f"--MSG-- Model saved to {ROOT_DIR_PROJECT}/{root_dir}/models/model.h5")
    #    
    #    #save the model
    #    model.save(os.path.join(ROOT_DIR_PROJECT, root_dir, "models", "model.h5"))
    #    print(f"--MSG-- Model saved to {ROOT_DIR_PROJECT}/{root_dir}/models/model.h5")
    #    
    #    #log the model
    #    mlflow.log_artifact(os.path.join(ROOT_DIR_PROJECT, root_dir, "models", "model.h5"))
    #    print(f"--MSG-- Model saved to {ROOT_DIR_PROJECT}/{root_dir}/models/model.h5")
    #    
    #    #log the model
    #    mlflow.log_artifact(os.path.join(ROOT_DIR_PROJECT, root_dir, "models", "model.h5"))
    #    print(f"--MSG-- Model saved to {ROOT_DIR_PROJECT}/{root_dir}/models/model.h5")
    #    mlflow.end_run()
#

if __name__ == "__main__":
    train_estimator_model(
        root_dir="yahoo",
        train_size= "189",
        lags= 5
    )