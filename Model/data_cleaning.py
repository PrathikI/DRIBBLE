import pandas as pd
import numpy as np

def preprocess_data(shot_logs_df):
    
    try:
        # Replace inf and -inf with NaN
        print("Checking for infinite values and replacing them with NaN...")
        shot_logs_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop all rows with any NaN values
        initial_count = len(shot_logs_df)
        shot_logs_df.dropna(inplace=True)
        final_count = len(shot_logs_df)
        print(f"Dropped {initial_count - final_count} rows due to NaN values.")
        print(f"{final_count} rows remaining after cleaning.")

        # Standardize text columns
        print("Standardizing text columns...")
        text_cols = ['TEAM_NAME', 'PLAYER_NAME', 'POSITION_GROUP', 'POSITION', 'HOME_TEAM', 'AWAY_TEAM', 'EVENT_TYPE', 'ACTION_TYPE', 'BASIC_ZONE', 'ZONE_NAME', 'ZONE_ABB']
        for col in text_cols:
            if col in shot_logs_df.columns:
                shot_logs_df[col] = shot_logs_df[col].str.upper()

        # Ensure proper data types
        print("Ensuring proper data types...")
        date_cols = ['GAME_DATE']
        for col in date_cols:
            if col in shot_logs_df.columns:
                shot_logs_df[col] = pd.to_datetime(shot_logs_df[col])

        # Check for any remaining NaN or infinite values
        print("Checking for any remaining NaN or infinite values...")
        if shot_logs_df.select_dtypes(include=[np.number]).apply(np.isinf).any().any():
            print("Warning: Infinite values detected after preprocessing.")
            shot_logs_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            shot_logs_df.dropna(inplace=True)

        print("Data preprocessing completed successfully.")
        return shot_logs_df

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return shot_logs_df  # Return the DataFrame even in case of error to maintain function integrity

