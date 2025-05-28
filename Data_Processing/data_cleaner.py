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
import numpy as np
from .metrics import Metrics
from configuration import Config


def clean_dataset() -> pd.DataFrame:
    """
    Load and clean the dataset with a universal missing-value strategy.

    Parameters:
    - input_csv: path to raw CSV file
    - output_csv: optional path to write cleaned CSV

    Returns:
    - Cleaned pandas DataFrame
    """
    #Class Loading
    metrics = Metrics()
    config = Config()

    #File Naming
    input_csv = config.data_path
    output_csv = config.output_path

    #Load Data
    df = pd.read_csv(input_csv)


    # 1. Parse Race_Time and extract components
    df['Race_Time'] = pd.to_datetime(df['Race_Time'], errors='coerce')
    df['year']      = df['Race_Time'].dt.year
    df['month']     = df['Race_Time'].dt.month
    df['dayofweek'] = df['Race_Time'].dt.dayofweek

    # 2. Convert imperial units to metric
    df = metrics.convert_to_metric(df)

    # 3. Handle missing values universally
    if config.missing_strategy == 'drop':
        # Drop any row with at least one missing value
        df = df.dropna()
    else:
        # Fill all missing values with a sentinel
        df = df.fillna('MISSING')

    # 4. Log-transform skewed numeric columns
    skew_cols = ['Prize', 'betfairSP', 'MarketOdds_PreviousRun', 'MarketOdds_2ndPreviousRun']
    for col in skew_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f'log_{col}'] = np.log1p(df[col].astype(float))
            df.drop(columns=[col], inplace=True)

    # 5. Output
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df