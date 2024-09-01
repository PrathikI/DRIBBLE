import pandas as pd
import os

def structure_output(analysis_results):
    structured_outputs = {}
    
    if isinstance(analysis_results, pd.DataFrame):
        structured_outputs['Analysis Results'] = analysis_results.reset_index(drop=True)
    elif isinstance(analysis_results, dict):
        for key, df in analysis_results.items():
            structured_outputs[key] = df.reset_index(drop=True)
    else:
        raise ValueError("Unexpected data structure for analysis_results. Expected a DataFrame or a dictionary of DataFrames.")
    
    return structured_outputs

def save_final_output(structured_outputs, cleaned_shot_logs_df):
    base_path = '/Users/iyeng1/Documents/Software-Development/PyDev II/DRIBBLE/OutputLogs/'
    
    # Save the cleaned shot logs to a CSV file
    cleaned_shots_path = os.path.join(base_path, 'cleaned_shot_logs.csv')
    print(f"Saving cleaned shot logs to CSV at: {cleaned_shots_path}")
    cleaned_shot_logs_df.to_csv(cleaned_shots_path, index=False)
    print("Cleaned shot logs saved successfully.")
    
    # Save the final model outputs to a single CSV file
    output_path = os.path.join(base_path, 'final_model_output.csv')
    print("Saving final model outputs...")
    
    with open(output_path, 'w') as f:
        for key, df in structured_outputs.items():
            f.write(f"\n\n# {key}\n")
            df.to_csv(f, index=False)
    
    print(f"Saving model output logs to CSV at: {output_path}")
    print("Model outputs saved successfully.")
