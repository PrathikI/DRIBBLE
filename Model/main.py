from data_loader import load_data
from data_cleaning import preprocess_data
from data_analysis import analyze_data
from model import train_and_evaluate_model
from output import structure_output, save_final_output

def main():
    print("Starting the NBA Player Prediction script...")
    
    base_path = '/Users/iyeng1/Documents/Software-Development/PyDev II/DRIBBLE/Data'
    
    print("Loading data...")
    shot_logs_df, game_schedule, player_stats = load_data(base_path)
    
    print("Preprocessing data...")
    shot_logs_df, player_stats, game_schedule = preprocess_data(shot_logs_df, player_stats, game_schedule)
    
    print("Analyzing data...")
    analysis_results = analyze_data(shot_logs_df)
    
    print("Training and evaluating the model...")
    model_results = train_and_evaluate_model(shot_logs_df)
    
    print("Model accuracies for different techniques and metrics:")
    for technique, accuracy in model_results.items():
        print(f"{technique.capitalize()} Metric: {accuracy:.2f}%")
    
    print("Structuring the output...")
    final_output = structure_output(analysis_results)
    
    print("Saving the final output...")
    save_final_output(final_output)

if __name__ == "__main__":
    main()
