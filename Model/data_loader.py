import pandas as pd
import os

def load_data(base_path):
    shot_logs = []
    for team_file in ['shot log ATL.csv', 'shot log BOS.csv', 'shot log BRO.csv', 'shot log CHA.csv,',
                    'shot log CHI.csv', 'shto log CLE.csv', 'shot log DAL.csv', 'shot log DEN.csv', 
                    'shot log DET.csv,', 'shot log GSW.csv', 'shot log HOU.csv', 'shot log IND.csv',
                    'shot log LAC.csv', 'shot log LAL.csv', 'shot log MEM.csv', 'shot log MIA.csv',
                    'shot log MIL.csv', 'shot log MIN.csv', 'shot log NOP.csv', 'shot log NYK.csv',
                    'shot log OKL.csv', 'shot log ORL.csv', 'shot log PHI.csv', 'shot log PHX.csv',
                    'shot log POR.csv', 'shot log SAC.csv', 'shot log SAS.csv', 'shot log TOR.csv',
                    'shot log UTA.csv', 'shot log WAS.csv']:
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
