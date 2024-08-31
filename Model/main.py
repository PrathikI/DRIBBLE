from data_loader import load_data
from data_cleaning import preprocess_data
from model import train_and_evaluate_model

def main():
    print("Starting the NBA Player Prediction script...")
    
    base_path = '/Users/iyeng1/Documents/Software-Development/PyDev II/DRIBBLE/Data/'
    
    print("Loading data...")
    shot_logs_df, game_schedule, player_stats = load_data(base_path)
    
    print("Preprocessing data...")
    shot_logs_df, player_stats, game_schedule = preprocess_data(shot_logs_df, player_stats, game_schedule)
    
    print("Training and evaluating the model...")
    results_df = train_and_evaluate_model(shot_logs_df)
    
    final_output_path = '/Users/iyeng1/Documents/Software-Development/PyDev II/DRIBBLE/OutputLogs/final_output.csv'
    print("Saving prediction and actual outcome comparison to CSV...")
    results_df.to_csv(final_output_path, index=False)
    print(f"Final output saved to {final_output_path}")

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
