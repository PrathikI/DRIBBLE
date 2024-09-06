import pandas as pd

def load_data(uploaded_file):
    """
    Loads the data from an uploaded file. Supports CSV and Excel formats.
    
    Args:
    uploaded_file (UploadedFile): The uploaded file object from Streamlit.

    Returns:
    pd.DataFrame: The loaded shot logs DataFrame.
    """
    if uploaded_file is not None:
        # Get the file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Check the file extension and load the appropriate format
        if file_extension == 'csv':
            shot_logs_df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            shot_logs_df = pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return shot_logs_df
    else:
        raise ValueError("No file uploaded.")
