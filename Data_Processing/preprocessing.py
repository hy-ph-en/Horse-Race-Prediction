from .data_cleaner import clean_dataset
from .feature_engineering import feature_engineering
from configuration import Config
import os
import pandas as pd

class Preprocessing():

    def __init__(self):
        self.config = Config()

    def preprocess_data(self, file_path):

        #Data Cleaning
        data = clean_dataset(file_path)

        #Data Feature Engineering
        data = feature_engineering(data)

        return data

    def check_and_preprocess(self):

        #Output Path
        file_path_train = self.config.data_path
        file_path_test = self.config.test_data_path

        output_path_train = file_path_train.replace('.csv', '_clean.csv')
        output_path_test = file_path_test.replace('.csv', '_clean.csv')

        if not os.path.exists(output_path_train) or not os.path.exists(output_path_test):
            print(f"{output_path_train} or {output_path_test} not found. Running preprocessing...")

            #Preprocess Data - Train
            file_path = self.config.data_path
            processed_data_train = self.preprocess_data(file_path)

            #Preprocess Data - Test
            file_path = self.config.test_data_path
            processed_data_test = self.preprocess_data(file_path)

            # Save the processed data to output_path
            processed_data_train.to_csv(output_path_train, index=False)
            processed_data_test.to_csv(output_path_test, index=False)
            
            return (processed_data_train, processed_data_test)
        else:
            print(f"Data already exists. Skipping preprocessing.")
            return (pd.read_csv(output_path_train), pd.read_csv(output_path_test))

