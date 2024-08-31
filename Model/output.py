import pandas as pd

def structure_output(analysis_results):
    # Assuming some processing to format the output
    formatted_output = analysis_results.reset_index()
    print("Output structured successfully.")
    return formatted_output

def save_final_output(final_output):
    output_path = '/Users/iyeng1/Documents/Software-Development/PyDev II/NBAShotPrediction/final_model_output.csv'
    final_output.to_csv(output_path, index=False)
    print(f"Final model output saved to {output_path}")
