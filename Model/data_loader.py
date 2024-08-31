import pandas as pd
import os

def load_data(base_path):
    shot_logs = []
    for team_file in ['shot log ATL.csv', 'shot log BOS.csv', 'shot log BRO.csv']:
        file_path = os.path.join(base_path, team_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            shot_logs.append(df)
    shot_logs_df = pd.concat(shot_logs, ignore_index=True)

    player_stats_path = os.path.join(base_path, 'Player Regular 16-17 Stats.csv')
    player_stats = pd.read_csv(player_stats_path)

    schedule_path = os.path.join(base_path, 'Game Schedule 16-17-Regular.csv')
    game_schedule = pd.read_csv(schedule_path)

    return shot_logs_df, game_schedule, player_stats
