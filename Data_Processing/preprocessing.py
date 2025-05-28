from .data_cleaner import clean_dataset
from .feature_engineering import feature_engineering
from configuration import Config
import os
import pandas as pd

class Preprocessing():

    def __init__(self):
        self.config = Config()
        self.output_path = 'Data/trainData_clean.csv'

    def preprocess_data(self):

        #Data Cleaning
        data = clean_dataset()

        #Data Feature Engineering
        data = feature_engineering(data)


        print(data)

        return data

    def check_and_preprocess(self):
        if not os.path.exists(self.output_path):
            print(f"{self.output_path} not found. Running preprocessing...")
            processed_data = self.preprocess_data()
            # Save the processed data to output_path
            processed_data.to_csv(self.output_path, index=False)
            return processed_data
        else:
            print(f"{self.output_path} already exists. Skipping preprocessing.")
            return pd.read_csv(self.output_path)

