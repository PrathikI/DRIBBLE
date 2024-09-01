import pandas as pd
import os

def load_data(base_path):
    shot_logs = []
    # Load only the 2004 dataset for now
    file_name = 'NBA_2004_Shots.csv'
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        shot_logs.append(df)
    
    # Combine all yearly data into a single DataFrame
    if shot_logs:  
        shot_logs_df = pd.concat(shot_logs, ignore_index=True)
    else:
        shot_logs_df = pd.DataFrame()  
    
    return shot_logs_df

# Uncomment the following lines when ready to process more years:
# for year in range(2005, 2025):  # From 2005 to 2024 inclusive
#     file_name = f'NBA_{year}_Shots.csv'
#     file_path = os.path.join(base_path, file_name)
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
#         shot_logs.append(df)
