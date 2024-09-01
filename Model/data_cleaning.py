import pandas as pd
import numpy as np

def preprocess_data(shot_logs_df, player_stats, game_schedule):
    print("Starting data preprocessing...")
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

        # Standardize text columns to proper case using .loc[] to avoid SettingWithCopyWarning
        print("Standardizing text columns...")
        shot_logs_df.loc[:, 'shoot player'] = shot_logs_df['shoot player'].str.title()
        shot_logs_df.loc[:, 'home team'] = shot_logs_df['home team'].str.upper()
        shot_logs_df.loc[:, 'current shot outcome'] = shot_logs_df['current shot outcome'].str.upper()
        shot_logs_df.loc[:, 'shot type'] = shot_logs_df['shot type'].str.title()

        # Check for any remaining NaN or infinite values
        print("Checking for any remaining NaN or infinite values...")
        if shot_logs_df.select_dtypes(include=[np.number]).apply(np.isinf).any().any():
            print("Warning: Infinite values detected after preprocessing.")
            shot_logs_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            shot_logs_df.dropna(inplace=True)

        print("Data preprocessing completed successfully.")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        # Ensure to return all three even in case of an error to maintain the expected function signature
        return shot_logs_df, player_stats, game_schedule

    # Save the cleaned dataset to a CSV file, overwrite if it already exists
    cleaned_file_path = '/Users/iyeng1/Documents/Software-Development/PyDev II/DRIBBLE/OutputLogs/cleaned_shot_logs.csv'
    shot_logs_df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned dataset saved at: {cleaned_file_path}")

    return shot_logs_df, player_stats, game_schedule
