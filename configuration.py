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
        #Feature Columns - SAFE FEATURES ONLY (No Data Leakage)
        self.feature_cols = [
            # Core engineered features (20)
            'Speed_PreviousRun_diff', 'Speed_PreviousRun_rank',
            'TrainerRating_diff', 'TrainerRating_rank',
            'JockeyRating_diff',  'JockeyRating_rank',
            'Age_diff', 'Age_rank', 'daysSinceLastRun_diff', 'daysSinceLastRun_rank',
            'SireRating_diff', 'SireRating_rank', 'DamsireRating_diff', 'DamsireRating_rank',
            'meanRunners_diff', 'meanRunners_rank',
            'MarketOdds_PreviousRun_diff', 'MarketOdds_PreviousRun_rank',
            'MarketOdds_2ndPreviousRun_diff', 'MarketOdds_2ndPreviousRun_rank',
            
            # Safe additional features (7)
            'Age_sq',                   # Non-linear age performance curve (numeric)
            'optimal_layoff',           # Racing-specific optimal rest period (binary) - uses daysSinceLastRun only
            'perf_jump',                # Speed improvement between last two runs - uses previous speeds only
            'SpeedPrev_rank',           # Previous run relative speed position - ranking of historical data
            'Horse_won_course_before',  # Historical course success - uses .shift(1) to exclude current race
            'Jockey_Trainer_pair',      # Partnership synergy effects - hash of names known pre-race
            'combined_rating_score',    # Synthesized multi-factor rating - based on external ratings only
            
            # Safe categorical features (5)
            'going_category',           # Track conditions (fast/good/soft) - known pre-race
            'season',                   # Seasonal racing patterns - based on race date
            'layoff_category',          # Rest period categories - based on daysSinceLastRun
            'time_category',            # Time of day - based on race time
            'field_size_category'       # Field size competitive effects - Runners count finalized pre-race
        ]

        'Holding for adding and replacement for Feature Columns during modification'
        #All Feature Columns - SAFE FEATURES ONLY (No Data Leakage)
        self.all_feature_cols = [
            # Currently used engineered features (20)
            'Speed_PreviousRun_diff', 'Speed_PreviousRun_rank',
            'TrainerRating_diff', 'TrainerRating_rank',
            'JockeyRating_diff',  'JockeyRating_rank',
            'Age_diff', 'Age_rank', 'daysSinceLastRun_diff', 'daysSinceLastRun_rank',
            'SireRating_diff', 'SireRating_rank', 'DamsireRating_diff', 'DamsireRating_rank',
            'meanRunners_diff', 'meanRunners_rank',
            'MarketOdds_PreviousRun_diff', 'MarketOdds_PreviousRun_rank',
            'MarketOdds_2ndPreviousRun_diff', 'MarketOdds_2ndPreviousRun_rank',
            
            # Metrics class features - SAFE ONLY (9)
            'distance_m', 'Speed_PreviousRun_mps', 'Speed_2ndPreviousRun_mps',
            'SpeedPrev_rank', 'Layoff_rank',
            'perf_jump',  # Speed improvement between last two runs
            'Jockey_Trainer_pair', 'Age_sq', 'Age_x_dist',
            # NOTE: Horse_won_course_before and Horse_won_going_before use .shift(1) so may be safe
            'Horse_won_course_before', 'Horse_won_going_before',
            
            # Temporal features (9)
            'race_hour', 'race_day_of_week', 'race_month', 'race_year',
            'time_category', 'is_weekend', 'season',
            'layoff_category', 'optimal_layoff',
            
            # Statistical features - historical only (13)
            'horse_career_wins', 'horse_career_runs', 'horse_win_rate',
            'horse_recent_form', 'horse_position_consistency',
            'horse_course_runs', 'horse_course_wins', 'horse_course_win_rate',
            'course_experience',
            'trainer_win_rate', 'trainer_strike_rate',
            'jockey_win_rate', 'jockey_strike_rate',
            
            # Performance features - safe only (4)
            'speed_improvement', 'speed_improving',
            'speed_prev_rank', 'speed_category',
            
            # Rating features (11)
            'trainer_rating_rank', 'high_rated_trainer',
            'jockey_rating_rank', 'high_rated_jockey',
            'sire_rating_rank', 'good_breeding',
            'TrainerRating_normalized', 'JockeyRating_normalized', 'SireRating_normalized',
            'combined_rating_score', 'combined_rating_rank',
            
            # Racing-specific features - safe only (11)
            'field_size_category', 'position_pct', 'field_size_advantage',
            'nmfp_category', 'nmfp_improvement', 'nmfp_improving',
            'distance_furlongs', 'distance_category',
            'prize_rank_by_date', 'high_value_race', 'going_category'
        ]
        
        # Features excluded due to data leakage (for reference)
        self.excluded_leakage_features = [
            # Market wisdom using current race odds
            'implied_prob', 'log_odds', 'delta_odds',
            # Form/momentum using current race data
            'speed_roll_mean', 'speed_roll_trend', 'recency_place_pct',
            # Current race betting and results
            'betfairSP_diff', 'betfairSP_rank', 'OddsRank', 'IsFavourite',
            'Position_diff', 'Position_rank', 'pdsBeaten_diff', 'pdsBeaten_rank',
            'NMFP_diff', 'NMFP_rank',
            # Current race performance
            'time_rank', 'time_behind_winner', 'fast_time',
            'beaten_lengths_rank', 'close_finish', 'margin_category'
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

