import pandas as pd
import numpy as np

def preprocess_data(shot_logs_df, return_metrics=False):
    try:
        initial_count = len(shot_logs_df)

        # Replace inf and -inf with NaN
        shot_logs_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop all rows with any NaN values
        shot_logs_df.dropna(inplace=True)
        final_count = len(shot_logs_df)

        # Standardize text columns
        shot_logs_df['PLAYER_NAME'] = shot_logs_df['PLAYER_NAME'].str.title()
        shot_logs_df['HOME_TEAM'] = shot_logs_df['HOME_TEAM'].str.upper()
        shot_logs_df['EVENT_TYPE'] = shot_logs_df['EVENT_TYPE'].str.upper()
        shot_logs_df['SHOT_TYPE'] = shot_logs_df['SHOT_TYPE'].str.title()

        # Optional: Return metrics
        if return_metrics:
            metrics = {
                'initial_count': initial_count,
                'final_count': final_count,
                'rows_dropped': initial_count - final_count,
                'null_values': shot_logs_df.isnull().sum().sum()
            }
            return shot_logs_df, metrics

        return shot_logs_df

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return shot_logs_df
