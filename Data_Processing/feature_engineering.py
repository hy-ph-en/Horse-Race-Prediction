import pandas as pd
import numpy as np
from .metrics import Metrics


def feature_engineering(data):
    """
    Comprehensive feature engineering pipeline for horse racing prediction model.
    Designed to work with data that has already been processed by data_cleaner.py
    
    Args:
        data (pd.DataFrame): Cleaned horse racing data from data_cleaner
        
    Returns:
        pd.DataFrame: Processed data with engineered features
    """
    # Initialize metrics class
    metrics = Metrics()
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    # Ensure proper data types for key columns
    df = _prepare_data_types(df)
    
    # Apply metric conversions (imperial to metric) - only if not already done
    if 'distanceYards' in df.columns and 'distance_m' not in df.columns:
        df = metrics.convert_to_metric(df)
    
    # Add within-race ranking features
    df = metrics.add_within_race_ranks(df)
    
    # Add form and momentum features
    df = metrics.add_form_momentum(df, window=3)
    
    # Add synergy and interaction features
    df = metrics.add_synergy_interactions(df)
    
    # Add market wisdom features
    df = metrics.add_market_wisdom(df)
    
    # Add calibration preparation features
    df = metrics.add_calibration_prep(df)
    
    # Add horse racing specific features
    df = _add_racing_specific_features(df)
    
    # Add temporal features (if Race_Time exists)
    df = _add_temporal_features(df)
    
    # Add statistical aggregations
    df = _add_statistical_features(df)
    
    # Add advanced performance metrics
    df = _add_performance_metrics(df)
    
    # Add rating-based features
    df = _add_rating_features(df)
    
    # Handle remaining missing values
    df = _handle_missing_values(df)
    
    return df


def _prepare_data_types(df):
    """Prepare data types for columns that might need conversion."""
    # Convert date columns if they exist and aren't already datetime
    if 'Race_Time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Race_Time']):
        df['Race_Time'] = pd.to_datetime(df['Race_Time'], errors='coerce')
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['Age', 'betfairSP', 'Position', 'timeSecs', 'daysSinceLastRun', 
                      'Speed_PreviousRun', 'Speed_2ndPreviousRun', 'NMFP', 'NMFPLTO',
                      'TrainerRating', 'JockeyRating', 'SireRating', 'DamsireRating',
                      'Runners', 'meanRunners', 'pdsBeaten']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def _add_racing_specific_features(df):
    """Add horse racing domain-specific features based on actual data structure."""
    
    # Beaten lengths analysis
    if 'pdsBeaten' in df.columns:
        # Beaten lengths rank within race
        df['beaten_lengths_rank'] = df.groupby('Race_ID')['pdsBeaten'].rank(ascending=True, method='dense')
        
        # Close finish indicator (beaten by less than 1 length)
        df['close_finish'] = (df['pdsBeaten'] <= 1.0).astype(int)
        
        # Margin categories
        df['margin_category'] = pd.cut(df['pdsBeaten'], 
                                     bins=[0, 0.5, 1.0, 2.0, 5.0, float('inf')], 
                                     labels=['nose', 'short_head', 'length', 'clear', 'well_beaten'])
    
    # Field size analysis
    if 'Runners' in df.columns:
        # Field size categories
        df['field_size_category'] = pd.cut(df['Runners'], 
                                         bins=[0, 6, 10, 16, float('inf')], 
                                         labels=['small', 'medium', 'large', 'very_large'])
        
        # Position as percentage of field
        if 'Position' in df.columns:
            df['position_pct'] = df['Position'] / df['Runners']
            
        # Field size advantage (smaller fields often easier to win)
        df['field_size_advantage'] = 1 / df['Runners']
    
    # NMFP (Next Meeting First Past) analysis
    if 'NMFP' in df.columns:
        # NMFP categories
        df['nmfp_category'] = pd.cut(df['NMFP'], 
                                   bins=[0, 0.2, 0.5, 0.8, 1.0], 
                                   labels=['poor', 'moderate', 'good', 'excellent'])
        
        # NMFP improvement from last time out
        if 'NMFPLTO' in df.columns:
            df['nmfp_improvement'] = df['NMFP'] - df['NMFPLTO']
            df['nmfp_improving'] = (df['nmfp_improvement'] > 0).astype(int)
    
    # Distance analysis
    if 'Distance' in df.columns:
        # Extract distance in furlongs for easier analysis
        df['distance_furlongs'] = df['Distance'].str.extract(r'(\d+)f').astype(float)
        
        # Distance categories
        df['distance_category'] = pd.cut(df['distance_furlongs'], 
                                       bins=[0, 6, 8, 12, float('inf')], 
                                       labels=['sprint', 'mile', 'middle', 'staying'])
    
    # Prize money analysis
    if 'Prize' in df.columns:
        # Prize money rank within date (class proxy)
        df['prize_rank_by_date'] = df.groupby(df['Race_Time'].dt.date)['Prize'].rank(ascending=False, method='dense')
        
        # High value race indicator
        df['high_value_race'] = (df['Prize'] > df['Prize'].quantile(0.75)).astype(int)
    
    # Going analysis
    if 'Going' in df.columns:
        # Going categories
        going_map = {
            'Firm': 'fast',
            'Good': 'good', 
            'Standard': 'good',
            'Soft': 'soft',
            'Heavy': 'soft'
        }
        df['going_category'] = df['Going'].map(going_map).fillna('good')
    
    return df


def _add_temporal_features(df):
    """Add time-based features using actual Race_Time column."""
    
    if 'Race_Time' in df.columns:
        # Extract time components
        df['race_hour'] = df['Race_Time'].dt.hour
        df['race_day_of_week'] = df['Race_Time'].dt.dayofweek
        df['race_month'] = df['Race_Time'].dt.month
        df['race_year'] = df['Race_Time'].dt.year
        
        # Time of day categories
        df['time_category'] = pd.cut(df['race_hour'], 
                                   bins=[0, 12, 15, 18, 24], 
                                   labels=['morning', 'afternoon', 'evening', 'night'])
        
        # Weekend racing
        df['is_weekend'] = (df['race_day_of_week'] >= 5).astype(int)
        
        # Season features
        df['season'] = df['race_month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
    
    # Days since last run features
    if 'daysSinceLastRun' in df.columns:
        # Layoff categories
        df['layoff_category'] = pd.cut(df['daysSinceLastRun'], 
                                     bins=[0, 14, 30, 60, 180, float('inf')], 
                                     labels=['fresh', 'recent', 'moderate', 'long', 'very_long'])
        
        # Optimal layoff (7-21 days often considered optimal)
        df['optimal_layoff'] = ((df['daysSinceLastRun'] >= 7) & 
                               (df['daysSinceLastRun'] <= 21)).astype(int)
    
    return df


def _add_statistical_features(df):
    """Add statistical aggregation features based on actual data."""
    
    # Horse career statistics
    if 'Horse' in df.columns and 'Position' in df.columns:
        # Career win rate
        df['horse_career_wins'] = df.groupby('Horse')['Position'].transform(lambda x: (x == 1).sum())
        df['horse_career_runs'] = df.groupby('Horse')['Position'].transform('count')
        df['horse_win_rate'] = df['horse_career_wins'] / df['horse_career_runs']
        
        # Recent form (last 5 runs)
        df['horse_recent_form'] = df.groupby('Horse')['Position'].transform(
            lambda x: (x.tail(5) <= 3).mean() if len(x) >= 5 else (x <= 3).mean()
        )
        
        # Consistency (standard deviation of positions)
        df['horse_position_consistency'] = df.groupby('Horse')['Position'].transform('std')
    
    # Course-specific performance
    if 'Course' in df.columns and 'Horse' in df.columns:
        # Course win rate
        df['horse_course_runs'] = df.groupby(['Horse', 'Course'])['Position'].transform('count')
        df['horse_course_wins'] = df.groupby(['Horse', 'Course'])['Position'].transform(lambda x: (x == 1).sum())
        df['horse_course_win_rate'] = df['horse_course_wins'] / df['horse_course_runs']
        
        # Course experience
        df['course_experience'] = df['horse_course_runs'] >= 3
    
    # Trainer and Jockey statistics
    if 'Trainer' in df.columns:
        df['trainer_win_rate'] = df.groupby('Trainer')['Position'].transform(lambda x: (x == 1).mean())
        df['trainer_strike_rate'] = df.groupby('Trainer')['Position'].transform(lambda x: (x <= 3).mean())
    
    if 'Jockey' in df.columns:
        df['jockey_win_rate'] = df.groupby('Jockey')['Position'].transform(lambda x: (x == 1).mean())
        df['jockey_strike_rate'] = df.groupby('Jockey')['Position'].transform(lambda x: (x <= 3).mean())
    
    return df


def _add_performance_metrics(df):
    """Add advanced performance metrics."""
    
    # Speed figures analysis
    if 'Speed_PreviousRun' in df.columns:
        # Speed improvement
        if 'Speed_2ndPreviousRun' in df.columns:
            df['speed_improvement'] = df['Speed_PreviousRun'] - df['Speed_2ndPreviousRun']
            df['speed_improving'] = (df['speed_improvement'] > 0).astype(int)
        
        # Speed rank within race
        df['speed_prev_rank'] = df.groupby('Race_ID')['Speed_PreviousRun'].rank(ascending=False, method='dense')
        
        # Speed categories
        df['speed_category'] = pd.qcut(df['Speed_PreviousRun'], 
                                     q=5, labels=['very_slow', 'slow', 'average', 'fast', 'very_fast'])
    
    # Time analysis
    if 'timeSecs' in df.columns:
        # Time rank within race
        df['time_rank'] = df.groupby('Race_ID')['timeSecs'].rank(ascending=True, method='dense')
        
        # Time relative to winner
        df['time_behind_winner'] = df.groupby('Race_ID')['timeSecs'].transform(lambda x: x - x.min())
        
        # Fast time indicator (within 2 seconds of winner)
        df['fast_time'] = (df['time_behind_winner'] <= 2.0).astype(int)
    
    return df


def _add_rating_features(df):
    """Add features based on rating columns."""
    
    # Trainer rating analysis
    if 'TrainerRating' in df.columns:
        df['trainer_rating_rank'] = df.groupby('Race_ID')['TrainerRating'].rank(ascending=False, method='dense')
        df['high_rated_trainer'] = (df['TrainerRating'] > df['TrainerRating'].quantile(0.75)).astype(int)
    
    # Jockey rating analysis
    if 'JockeyRating' in df.columns:
        df['jockey_rating_rank'] = df.groupby('Race_ID')['JockeyRating'].rank(ascending=False, method='dense')
        df['high_rated_jockey'] = (df['JockeyRating'] > df['JockeyRating'].quantile(0.75)).astype(int)
    
    # Sire rating analysis
    if 'SireRating' in df.columns:
        df['sire_rating_rank'] = df.groupby('Race_ID')['SireRating'].rank(ascending=False, method='dense')
        df['good_breeding'] = (df['SireRating'] > df['SireRating'].quantile(0.6)).astype(int)
    
    # Combined rating score
    rating_cols = ['TrainerRating', 'JockeyRating', 'SireRating']
    available_ratings = [col for col in rating_cols if col in df.columns]
    
    if available_ratings:
        # Normalize ratings and create combined score
        for col in available_ratings:
            df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        normalized_cols = [f'{col}_normalized' for col in available_ratings]
        df['combined_rating_score'] = df[normalized_cols].mean(axis=1)
        df['combined_rating_rank'] = df.groupby('Race_ID')['combined_rating_score'].rank(ascending=False, method='dense')
    
    return df


def _handle_missing_values(df):
    """Handle missing values with domain-appropriate strategies."""
    
    # Fill numeric features with appropriate values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            # Use median for most numeric columns
            if col in ['Age', 'daysSinceLastRun', 'Runners']:
                df[col] = df[col].fillna(df[col].median())
            # Use 0 for rating/performance columns that might legitimately be missing
            elif 'rating' in col.lower() or 'speed' in col.lower():
                df[col] = df[col].fillna(0)
            # Use group median for race-specific features
            elif col in ['betfairSP', 'timeSecs']:
                df[col] = df.groupby('Race_ID')[col].transform(lambda x: x.fillna(x.median()))
                df[col] = df[col].fillna(df[col].median())  # Fill any remaining
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_val)
    
    return df
