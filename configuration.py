'Configuration file for the project'
class Config():

    def __init__(self):
        #Paths to Data
        self.data_path = 'Data/trainData.csv'
        self.test_data_path = 'Data/testData.csv'
        
        #Config for Data Cleaning
        self.missing_strategy = 'drop'

        #Varry On Desired Training Time  
        self.data_incusion = 1      #Legacy 


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

        'Model Calibration and Scaling Configuration'
        # Softmax scaling factor for final probability normalization
        # Higher values = more aggressive normalization (sharper probability distributions)
        # Lower values = less aggressive normalization (softer probability distributions)  
        # Range: 1-10, Current optimal: 6
        self.softmax_scale_factor = 6
        
        # Meta-model calibration settings
        # Set use_meta_calibration=True to apply post-hoc calibration to meta-model predictions
        # This can help with over/under-confident predictions but may reduce discrimination
        self.use_meta_calibration = False  # Whether to apply calibration to meta-model
        self.calibration_method = 'isotonic'  # 'isotonic' or 'sigmoid'
        self.calibration_cv_folds = 3  # Cross-validation folds for calibration
        
        # Alternative scaling methods (experimental)
        # Set use_alternative_scaling=True to use weighted normalization instead of softmax
        # This preserves more of the original signal but may not sum to 1 perfectly
        self.use_alternative_scaling = False  # Use weighted normalization instead of softmax
        self.epsilon_scaling = 1e-8  # Epsilon for weighted normalization
        
        
        # Diagnostic and Testing Configuration
        self.show_detailed_diagnostics = False  # Whether to show detailed scale factor testing and diagnostics

