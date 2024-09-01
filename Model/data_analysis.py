def analyze_data(shot_logs_df):
    print("Starting data analysis...")
    
    # Analyze shot success rates per player
    player_performance = shot_logs_df.pivot_table(index='PLAYER_NAME', columns='SHOT_MADE', values='GAME_ID', aggfunc='count', fill_value=0)
    player_performance.columns = ['Shots Missed', 'Shots Made']
    player_performance['Success Rate'] = (player_performance['Shots Made'] / (player_performance['Shots Made'] + player_performance['Shots Missed'])) * 100
    
    # Analyze shot type distributions
    shot_type_distribution = shot_logs_df.groupby(['SHOT_TYPE']).size().reset_index(name='Count')
    
    # Performance in different court zones
    zone_performance = shot_logs_df.pivot_table(index='BASIC_ZONE', columns='SHOT_MADE', values='GAME_ID', aggfunc='count', fill_value=0)
    zone_performance.columns = ['Shots Missed', 'Shots Made']
    zone_performance['Success Rate'] = (zone_performance['Shots Made'] / (zone_performance['Shots Made'] + zone_performance['Shots Missed'])) * 100

    print("Data analysis completed successfully.")
    # Combining all analysis into a single dictionary for output
    analysis_results = {
        'Player Performance': player_performance,
        'Shot Type Distribution': shot_type_distribution,
        'Zone Performance': zone_performance
    }

    return analysis_results
