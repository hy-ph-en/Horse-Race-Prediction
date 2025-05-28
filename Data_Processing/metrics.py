import pandas as pd
import numpy as np


class Metrics:
    def __init__(self):
        pass

    #Metric Standarisation - Preference and assumption that the model can handle flat conversions easier
    def convert_to_metric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert imperial units in the DataFrame to metric:
        - distanceYards -> distance_m (meters)
        - speed columns in yards/sec -> m/sec
        Drops original imperial columns.
        """
        # Distance conversion (yards -> meters)
        # 1 yard = 0.9144 meters
        if 'distanceYards' in df.columns:
            df['distance_m'] = df['distanceYards'] * 0.9144
            df.drop(columns=['Distance', 'distanceYards'], inplace=True, errors='ignore')

        # Speed conversion (yards/sec -> meters/sec)
        speed_cols = ['Speed_PreviousRun', 'Speed_2ndPreviousRun']
        for col in speed_cols:
            if col in df.columns:
                df[f'{col}_mps'] = df[col] * 0.9144
                df.drop(columns=[col], inplace=True)

        return df
    
    def add_within_race_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds within-race rank features:
        - SpeedPrev_rank: rank of Speed_PreviousRun_mps within each Race_ID
        - OddsRank: rank of betfairSP (or implied odds) within each Race_ID
        - IsFavourite: boolean flag for OddsRank == 1
        - Layoff_rank: rank of daysSinceLastRun within each Race_ID
        
        PLUS: Creates the _diff and _rank features expected by the configuration
        """
        df = df.copy()
        
        # Original features
        if 'Speed_PreviousRun_mps' in df.columns:
            df['SpeedPrev_rank'] = df.groupby('Race_ID')['Speed_PreviousRun_mps'] \
                                    .rank(ascending=False, method='dense')
        # Odds rank (using betfairSP if available)
        if 'betfairSP' in df.columns:
            df['OddsRank'] = df.groupby('Race_ID')['betfairSP'] \
                                .rank(ascending=True, method='dense')
            df['IsFavourite'] = (df['OddsRank'] == 1).astype(int)
        # Layoff rank
        if 'daysSinceLastRun' in df.columns:
            df['Layoff_rank'] = df.groupby('Race_ID')['daysSinceLastRun'] \
                                    .rank(ascending=False, method='dense')
        
        # NEW: Add the expected _diff and _rank features
        df = self._add_expected_features(df)
        
        return df

    def _add_expected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the specific _diff and _rank features expected by the configuration.
        _diff features: difference from race mean
        _rank features: within-race ranking
        """
        # Define the base columns and their expected feature names
        feature_mapping = {
            'Speed_PreviousRun_mps': 'Speed_PreviousRun',
            'TrainerRating': 'TrainerRating', 
            'JockeyRating': 'JockeyRating',
            'Age': 'Age',
            'daysSinceLastRun': 'daysSinceLastRun',
            'SireRating': 'SireRating',
            'DamsireRating': 'DamsireRating',
            'meanRunners': 'meanRunners',
            'log_betfairSP': 'betfairSP',  # Use log version if available
            'Position': 'Position',
            'pdsBeaten': 'pdsBeaten',
            'NMFP': 'NMFP',
            'log_MarketOdds_PreviousRun': 'MarketOdds_PreviousRun',
            'log_MarketOdds_2ndPreviousRun': 'MarketOdds_2ndPreviousRun'
        }
        
        # Create _diff and _rank features for each available column
        for source_col, feature_name in feature_mapping.items():
            if source_col in df.columns:
                # Create _diff feature (difference from race mean)
                race_mean = df.groupby('Race_ID')[source_col].transform('mean')
                df[f'{feature_name}_diff'] = df[source_col] - race_mean
                
                # Create _rank feature (within-race ranking)
                # For most features, higher is better (ascending=False)
                # For Position and pdsBeaten, lower is better (ascending=True)
                if feature_name in ['Position', 'pdsBeaten']:
                    df[f'{feature_name}_rank'] = df.groupby('Race_ID')[source_col].rank(ascending=True, method='dense')
                else:
                    df[f'{feature_name}_rank'] = df.groupby('Race_ID')[source_col].rank(ascending=False, method='dense')
        
        # Handle betfairSP if log version not available
        if 'betfairSP' in df.columns and 'log_betfairSP' not in df.columns:
            race_mean = df.groupby('Race_ID')['betfairSP'].transform('mean')
            df['betfairSP_diff'] = df['betfairSP'] - race_mean
            df['betfairSP_rank'] = df.groupby('Race_ID')['betfairSP'].rank(ascending=True, method='dense')  # Lower odds = better
        
        # Handle original MarketOdds columns if log versions not available
        for odds_col in ['MarketOdds_PreviousRun', 'MarketOdds_2ndPreviousRun']:
            if odds_col in df.columns and f'log_{odds_col}' not in df.columns:
                race_mean = df.groupby('Race_ID')[odds_col].transform('mean')
                df[f'{odds_col}_diff'] = df[odds_col] - race_mean
                df[f'{odds_col}_rank'] = df.groupby('Race_ID')[odds_col].rank(ascending=True, method='dense')  # Lower odds = better
        
        return df

    def add_form_momentum(self, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """
        Adds form and momentum features:
        - Rolling mean and trend of timeSecs over last `window` runs per horse
        - Recency-weighted place%: exp decay of finishing in top 3
        - Performance jump: difference between last two speeds
        """
        df = df.copy().sort_values(['Horse', 'Race_Time'])
        # Rolling mean and slope
        df['speed_roll_mean'] = df.groupby('Horse')['timeSecs'] \
                                    .rolling(window, min_periods=1) \
                                    .mean().reset_index(level=0, drop=True)
        # Linear trend (slope) using polyfit on rolling window
        def compute_slope(x):
            if len(x) < 2:
                return 0.0
            y = np.array(x)
            idx = np.arange(len(y))
            slope = np.polyfit(idx, y, 1)[0]
            return slope
        df['speed_roll_trend'] = df.groupby('Horse')['timeSecs'] \
                                    .rolling(window, min_periods=2) \
                                    .apply(compute_slope, raw=False) \
                                    .reset_index(level=0, drop=True)
        # Recency-weighted place%
        def decay_place(series, alpha=0.5):
            weights = alpha ** np.arange(len(series))[::-1]
            place = (series <= 3).astype(int)
            return np.dot(place, weights) / weights.sum()
        df['recency_place_pct'] = df.groupby('Horse')['Position'] \
                                    .rolling(window, min_periods=1) \
                                    .apply(lambda x: decay_place(x), raw=False) \
                                    .reset_index(level=0, drop=True)
        # Performance jump
        if 'Speed_PreviousRun_mps' in df.columns and 'Speed_2ndPreviousRun_mps' in df.columns:
            df['perf_jump'] = df['Speed_PreviousRun_mps'] - df['Speed_2ndPreviousRun_mps']
        return df


    def add_synergy_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds interaction features:
        - Jockey_Trainer_pair: hashed combination of Trainer and Jockey
        - Horse_Course and Horse_Going historical win flags
        - Age squared and Age * distance_m
        """
        df = df.copy()
        # Jockey-Trainer pair as hash
        if 'Trainer' in df.columns and 'Jockey' in df.columns:
            pair = df['Trainer'].astype(str) + '_' + df['Jockey'].astype(str)
            df['Jockey_Trainer_pair'] = pair.apply(lambda x: hash(x) % (10**8))
        # Horse-course and horse-going flags
        if 'Horse' in df.columns and 'Course' in df.columns:
            df['Horse_won_course_before'] = df.groupby(['Horse','Course'])['Position'] \
                                            .transform(lambda x: (x.shift(1) == 1).cummax().fillna(0))
        if 'Horse' in df.columns and 'Going' in df.columns:
            df['Horse_won_going_before'] = df.groupby(['Horse','Going'])['Position'] \
                                            .transform(lambda x: (x.shift(1) == 1).cummax().fillna(0))
        # Age interactions
        if 'Age' in df.columns and 'distance_m' in df.columns:
            df['Age_sq'] = df['Age'] ** 2
            df['Age_x_dist'] = df['Age'] * df['distance_m']
        return df


    def add_market_wisdom(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds market-based features:
        - implied_prob: 1 / betfairSP
        - log_odds: log1p of betfairSP
        - delta_odds: log(betfairSP) - log(MarketOdds_PreviousRun)
        """
        df = df.copy()
        if 'betfairSP' in df.columns:
            df['implied_prob'] = 1.0 / df['betfairSP'].replace({0: np.nan})
            df['log_odds'] = np.log1p(df['betfairSP'].astype(float))
        if 'MarketOdds_PreviousRun' in df.columns and 'betfairSP' in df.columns:
            df['delta_odds'] = np.log1p(df['betfairSP'].astype(float)) \
                                - np.log1p(df['MarketOdds_PreviousRun'].astype(float))
        return df


    def add_calibration_prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the target for probability models by adding a softmax-ready
        indicator and drop raw Position if needed.
        """
        df = df.copy()
        # Create win flag
        df['is_win'] = (df['Position'] == 1).astype(int)
        # Optionally drop Position if not needed
        # df.drop(columns=['Position'], inplace=True)
        return df
