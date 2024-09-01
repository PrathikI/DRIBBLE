from data_loader import load_data
from data_cleaning import preprocess_data
from model import train_and_evaluate_model
from output import structure_output, save_final_output

def main():
    print("Starting the NBA Player Prediction script...")
    
    base_path = '/Users/iyeng1/Documents/Software-Development/PyDev II/DRIBBLE/Data/'
    
    print("Loading data...")
    shot_logs_df = load_data(base_path)
    
    print("Starting data preprocessing...")
    shot_logs_df = preprocess_data(shot_logs_df)
    
    print("Training and evaluating the model...")
    results_df = train_and_evaluate_model(shot_logs_df)
    
    print("Structuring the final output...")
    structured_results = structure_output(results_df)  
    
    print("Output structured successfully.")
    save_final_output(structured_results, shot_logs_df)  

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
