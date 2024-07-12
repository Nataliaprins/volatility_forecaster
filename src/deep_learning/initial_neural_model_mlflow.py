" this function initializes the neural network model and loads the weights from the saved model with keras"
import glob
import os

from constants import ROOT_DIR_PROJECT
from src.mlflow.load_test_data_mlflow import load_test_data
from src.mlflow.load_train_data_mlflow import load_train_data


def initial_neural_model(root_dir, train_size, lags, n_features, n_steps):
     #obtain the train data from the data folder
    path_to_train_data= os.path.join(ROOT_DIR_PROJECT,
                                     root_dir,
                                     "processed",
                                     "train",
                                     "train_*.csv")
    data_train_files = glob.glob(path_to_train_data)

   #read the train data
    for data_file in data_train_files:
        x_train, y_train = load_train_data(
            root_dir= root_dir,
            train_size= train_size,
            lags= lags,
            stock_name= data_file.split('_')[-1].replace(".csv", "")
            )     
    #read the test data
        x_test, y_test = load_test_data(
            stock_name= data_file.split('_')[-1].replace(".csv", ""),
            root_dir= root_dir,
            train_size= train_size,
            lags= lags
            )   

        # inicialize the model
        n_features = 1
        n_steps = 3
        neural_model = Sequential()
        neural_model.add(LSTM(50, activation="relu", input_shape=(n_steps, n_features)))
        neural_model.add(Dense(1))
        neural_model.compile(optimizer="adam", loss="mse")



if __name__ == "__main__":
    initial_neural_model(root_dir="yahoo",
                         train_size=189,
                         lags=5,
                            n_features=1,
                            n_steps=3
                        )