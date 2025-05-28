'Configuration file for the project'
class Config():

    def __init__(self):
        #Paths to Data
        self.data_path = 'Data/trainData.csv'
        self.test_data_path = 'Data/testData.csv'
        
        #Config for Data Cleaning
        self.missing_strategy = 'drop'

        #Varry On Desired Training Time  
        self.data_incusion = 1


        'Can Change the Features Desired for Training'
        #Feature Columns - Removed post-race variables: betfairSP, Position, pdsBeaten, NMFP
        self.feature_cols = [
            'Speed_PreviousRun_diff', 'Speed_PreviousRun_rank',
            'TrainerRating_diff', 'TrainerRating_rank',
            'JockeyRating_diff',  'JockeyRating_rank',
            'Age_diff', 'Age_rank', 'daysSinceLastRun_diff', 'daysSinceLastRun_rank',
            'SireRating_diff', 'SireRating_rank', 'DamsireRating_diff', 'DamsireRating_rank',
            'meanRunners_diff', 'meanRunners_rank',
            'MarketOdds_PreviousRun_diff', 'MarketOdds_PreviousRun_rank',
            'MarketOdds_2ndPreviousRun_diff', 'MarketOdds_2ndPreviousRun_rank'
        ]

        'Holding for adding and replacement for Feature Columns during modification'
        #All Feature Columns
        self.all_feature_cols = [
            'Speed_PreviousRun_diff', 'Speed_PreviousRun_rank',
            'TrainerRating_diff', 'TrainerRating_rank',
            'JockeyRating_diff',  'JockeyRating_rank',
            'Age_diff', 'Age_rank', 'daysSinceLastRun_diff', 'daysSinceLastRun_rank',
            'SireRating_diff', 'SireRating_rank', 'DamsireRating_diff', 'DamsireRating_rank',
            'meanRunners_diff', 'meanRunners_rank', 'betfairSP_diff', 'betfairSP_rank',
            'Position_diff', 'Position_rank', 'pdsBeaten_diff', 'pdsBeaten_rank',
            'NMFP_diff', 'NMFP_rank', 'MarketOdds_PreviousRun_diff', 'MarketOdds_PreviousRun_rank',
            'MarketOdds_2ndPreviousRun_diff', 'MarketOdds_2ndPreviousRun_rank'
        ]

        'Can Modify the Original Columns to be Used'
        #All Columns - None Feature Engineering
        self.orignal_cols = [
        'Race_Time', 'Race_ID', 'Course', 'Distance', 'distanceYards', 'Prize', 
        'Going', 'Horse', 'Trainer', 'Jockey', 'betfairSP', 'Position', 'timeSecs', 
        'pdsBeaten', 'NMFP', 'Runners', 'Age', 'Speed_PreviousRun', 'Speed_2ndPreviousRun', 
        'NMFPLTO', 'MarketOdds_PreviousRun', 'MarketOdds_2ndPreviousRun', 'TrainerRating', 
        'JockeyRating', 'daysSinceLastRun', 'SireRating', 'DamsireRating', 'meanRunners'
        ]


