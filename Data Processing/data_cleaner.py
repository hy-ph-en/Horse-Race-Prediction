# clean_dataset.py
"""Utility for loading and cleaning the horse-racing training table into metric units.

This script performs:
  - Distance conversion to meters
  - Speed conversion to meters/second
  - Datetime parsing
  - Missing value imputation
  - Log-transform of skewed variables

Usage:
    from clean_dataset import clean_dataset
    df_clean = clean_dataset("trainData.csv")
"""
import pandas as pd
from metrics import Metrics


def clean_dataset(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    # Load raw data
    df = pd.read_csv(input_csv)
    # Create Metric Object
    metrics = Metrics()

    # --- 1. Date and time parsing ---
    df['Race_Time'] = pd.to_datetime(df['Race_Time'])
    df['year'] = df['Race_Time'].dt.year
    df['month'] = df['Race_Time'].dt.month
    df['dayofweek'] = df['Race_Time'].dt.dayofweek

    # --- 2. Metric unit conversion ---
    df = metrics.convert_to_metric(df)

    # --- 3. Missing value imputation ---
    # Numeric columns: median impute
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # Categorical columns: fill with 'MISSING'
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df[cat_cols] = df[cat_cols].fillna('MISSING')

    # --- 4. Log-transform skewed numeric columns ---
    skew_cols = ['Prize', 'betfairSP', 'MarketOdds_PreviousRun', 'MarketOdds_2ndPreviousRun']
    for col in skew_cols:
        if col in df.columns:
            df[f'log_{col}'] = df[col].apply(lambda x: pd.np.log1p(x))
            df.drop(columns=[col], inplace=True)

    # --- 5. Final tidy output ---
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df


if __name__ == '__main__':
    # Example usage
    df_clean = clean_dataset('trainData.csv', 'trainData_clean.csv')
    print("Cleaned data saved to trainData_clean.csv")
