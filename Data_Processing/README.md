# Horse Racing Feature Engineering Pipeline

This module provides a comprehensive feature engineering pipeline specifically designed for horse racing predictive models. It transforms cleaned racing data into a rich set of features that capture various aspects of horse racing performance.

## Overview

The feature engineering pipeline is designed to work with data that has already been processed by `data_cleaner.py` and applies multiple categories of transformations:

1. **Metric Conversions** - Convert imperial units to metric (if needed)
2. **Within-Race Rankings** - Rank horses within each race
3. **Form & Momentum** - Capture recent performance trends
4. **Synergy & Interactions** - Model relationships between entities
5. **Market Wisdom** - Extract insights from betting markets
6. **Racing-Specific Features** - Domain-specific horse racing features
7. **Temporal Features** - Time-based patterns
8. **Statistical Aggregations** - Historical performance metrics
9. **Performance Metrics** - Advanced speed and time analysis
10. **Rating Features** - Trainer, jockey, and sire rating analysis

## Usage

```python
from Data_Processing.feature_engineering import feature_engineering

# Apply to your cleaned data
engineered_data = feature_engineering(your_cleaned_data)
```

## Input Data Structure

The pipeline expects data with the following columns (from Example.csv):

### Core Columns
- `Race_Time` - Date/time of race
- `Race_ID` - Unique identifier for each race
- `Course` - Race course name
- `Distance` - Race distance (e.g., "6f 20y")
- `distanceYards` - Distance in yards
- `Prize` - Prize money
- `Going` - Track conditions
- `Horse` - Horse name/identifier
- `Trainer` - Trainer name
- `Jockey` - Jockey name
- `Position` - Finishing position
- `Runners` - Number of runners in race

### Performance Columns
- `betfairSP` - Betfair starting price
- `timeSecs` - Race time in seconds
- `pdsBeaten` - Lengths beaten
- `Speed_PreviousRun` - Previous run speed rating
- `Speed_2ndPreviousRun` - Second previous run speed rating
- `daysSinceLastRun` - Days since last race

### Form Columns
- `NMFP` - Next Meeting First Past rating
- `NMFPLTO` - NMFP Last Time Out
- `Age` - Horse age

### Rating Columns
- `TrainerRating` - Trainer performance rating
- `JockeyRating` - Jockey performance rating
- `SireRating` - Sire (father) rating
- `DamsireRating` - Damsire (maternal grandfather) rating

### Market Columns
- `MarketOdds_PreviousRun` - Previous run market odds
- `MarketOdds_2ndPreviousRun` - Second previous run market odds
- `meanRunners` - Average field size

## Generated Features

### Racing-Specific Features
- `beaten_lengths_rank` - Rank by lengths beaten within race
- `close_finish` - Flag for races decided by less than 1 length
- `margin_category` - Categorized winning margins
- `field_size_category` - Small/medium/large/very_large field sizes
- `position_pct` - Position as percentage of field size
- `field_size_advantage` - Inverse of field size
- `nmfp_category` - NMFP performance categories
- `nmfp_improvement` - Improvement in NMFP from last run
- `distance_furlongs` - Distance extracted in furlongs
- `distance_category` - Sprint/mile/middle/staying distances
- `going_category` - Simplified going conditions

### Temporal Features
- `race_hour`, `race_day_of_week`, `race_month`, `race_year` - Time components
- `time_category` - Morning/afternoon/evening/night
- `is_weekend` - Weekend racing flag
- `season` - Spring/summer/autumn/winter
- `layoff_category` - Fresh/recent/moderate/long/very_long layoffs
- `optimal_layoff` - 7-21 day optimal layoff flag

### Performance Metrics
- `speed_improvement` - Improvement from second last to last run
- `speed_improving` - Boolean flag for speed improvement
- `speed_prev_rank` - Speed ranking within race
- `speed_category` - Speed performance quintiles
- `time_rank` - Time ranking within race
- `time_behind_winner` - Seconds behind race winner
- `fast_time` - Flag for times within 2 seconds of winner

### Statistical Features
- `horse_win_rate` - Career win percentage
- `horse_recent_form` - Recent form (last 5 runs in top 3)
- `horse_position_consistency` - Standard deviation of positions
- `horse_course_win_rate` - Win rate at specific course
- `course_experience` - Flag for 3+ runs at course
- `trainer_win_rate` - Trainer overall win rate
- `trainer_strike_rate` - Trainer top-3 rate
- `jockey_win_rate` - Jockey overall win rate
- `jockey_strike_rate` - Jockey top-3 rate

### Rating Features
- `trainer_rating_rank` - Trainer rating rank within race
- `jockey_rating_rank` - Jockey rating rank within race
- `sire_rating_rank` - Sire rating rank within race
- `high_rated_trainer/jockey` - Top quartile rating flags
- `good_breeding` - Top 60% sire rating flag
- `combined_rating_score` - Normalized combined rating
- `combined_rating_rank` - Combined rating rank within race

### Market Features
- `implied_prob` - Implied probability from odds
- `log_odds` - Log-transformed odds
- `delta_odds` - Change in odds from previous run
- `OddsRank` - Odds ranking within race
- `IsFavourite` - Race favorite flag

## Testing

Run the test script to see the pipeline in action:

```python
python test_feature_engineering.py
```

This will:
- Load the Example.csv data
- Apply feature engineering
- Show statistics and new features created
- Save the engineered data

## Key Improvements for Horse Racing

1. **Beaten Lengths Analysis** - Critical for understanding race margins
2. **NMFP Integration** - Uses Next Meeting First Past ratings
3. **Field Size Effects** - Accounts for competitive field sizes
4. **Layoff Optimization** - Identifies optimal rest periods
5. **Course Specialization** - Tracks course-specific performance
6. **Rating Synthesis** - Combines trainer, jockey, and breeding ratings
7. **Speed Progression** - Tracks improvement trends
8. **Market Intelligence** - Extracts wisdom from betting markets

## Performance Considerations

- Optimized for the specific data structure in Example.csv
- Handles missing values appropriately for each feature type
- Uses vectorized operations for efficiency
- Avoids data leakage in time-series context
- Creates 50+ new features from the original 28 columns

## Dependencies

- pandas
- numpy

## File Structure

- `feature_engineering.py` - Main feature engineering pipeline
- `metrics.py` - Metrics class with specialized transformations
- `test_feature_engineering.py` - Test script with Example.csv
- `README.md` - This documentation 