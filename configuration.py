'Configuration file for the project'
class Config():

    def __init__(self):
        #Paths to Data
        self.data_path = 'Data/trainData.csv'
        self.test_data_path = 'Data/testData.csv'
        self.output_path = 'Data/trainData_clean.csv'
        
        #Config for Data Cleaning
        self.missing_strategy = 'fill'

        #Varry On Desired Training Time  
        self.data_incusion = 1

        #Feature Columns
        self.feature_cols = [
            'Speed_PreviousRun_diff', 'Speed_PreviousRun_rank',
            'TrainerRating_diff', 'TrainerRating_rank',
            'JockeyRating_diff',  'JockeyRating_rank',
            'Age_diff', 'Age_rank', 'daysSinceLastRun_diff', 'daysSinceLastRun_rank'
        ]



