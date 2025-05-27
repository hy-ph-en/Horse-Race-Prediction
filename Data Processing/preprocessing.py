from data_cleaner import clean_dataset
from feature_engineering import feature_engineering
from configuration import Config

class Preprocessing():

    def __init__(self):
        self.config = Config()

    def preprocess_data(self, data):

        #Data Cleaning
        data = self.data_cleaner.clean_data(data)

        #Data Feature Engineering
        data = feature_engineering(data)


        return data

