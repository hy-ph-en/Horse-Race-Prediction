import pandas as pd


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