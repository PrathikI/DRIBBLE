import sys
import os

# Ensure the parent directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from Model.data_loader import load_data
from Model.data_cleaning import preprocess_data
from Model.model import train_and_evaluate_model
from Model.output import structure_output, save_final_output
from FrontEnd.descriptions import *  # Import descriptions

# Customizing the look and feel with CSS for font size and menu bar
st.set_page_config(page_title="D.R.I.B.B.L.E", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .reportview-container .markdown-text-container {
        font-size: 20px; /* Increased font size for the descriptions */
    }
    .sidebar .sidebar-content {
        font-size: 20px;
    }
    .reportview-container .main .block-container{
        max-width: 80%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [role="tab"] {
        font-size: 22px; /* Increased font size for the nav bar */
        padding-left: 20px; /* Increase spacing between text */
        margin-right: 15px; /* Increase space between tabs */
    }
    h3 {
        color: var(--primary-color); /* Use the same red as the tab underline for the .py file steps */
    }
    .data-metrics {
        font-size: 18px;
        color: #FFFFFF;
        margin-top: 15px;
    }
    .data-metrics-value {
        font-size: 18px;
        color: #007ACC;
    }
    .download-button {
        display: flex;
        justify-content: flex-end;
        margin-top: -30px;
        margin-bottom: 10px;
    }
    .cleaned-dataset-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 60%;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Home", 
    "Data Ingestion", 
    "Data Transformation", 
    "Model Development and Training", 
    "Model Performance Analysis"
])

with tab1:
    st.markdown(home_description, unsafe_allow_html=True)
    st.markdown(project_structure, unsafe_allow_html=True)
    
    st.markdown(data_loader_description, unsafe_allow_html=True)
    st.markdown(data_cleaning_description, unsafe_allow_html=True)
    st.markdown(data_analysis_description, unsafe_allow_html=True)
    st.markdown(model_description, unsafe_allow_html=True)
    st.markdown(output_description, unsafe_allow_html=True)
    
    st.markdown(running_project, unsafe_allow_html=True)
    st.markdown(model_accuracies, unsafe_allow_html=True)
    
    if 'shot_logs_df' in locals():
        st.subheader("Full Dataset Preview")
        st.dataframe(shot_logs_df, height=500)

with tab2:
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file is not None:
        shot_logs_df = pd.read_csv(uploaded_file)
        st.success("Data Ingested Successfully!")
        st.write(shot_logs_df.head())

with tab3:
    st.header("Data Transformation")
    if 'shot_logs_df' in locals():
        shot_logs_df, metrics = preprocess_data(shot_logs_df, return_metrics=True)
        st.success("Data Transformed Successfully!")

        # Display cleaned dataset header with download button aligned to the right
        st.markdown('<div class="cleaned-dataset-header">', unsafe_allow_html=True)
        st.subheader("Cleaned Dataset")
        st.download_button(
            label="Download cleaned data",
            data=shot_logs_df.to_csv(index=False),
            file_name='cleaned_shot_logs.csv',
            mime='text/csv',
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Add ability to sort and filter the dataframe
        st.dataframe(shot_logs_df)

        # Display metrics about the cleaning process below the table with bullet points
        st.subheader("Data Cleaning Metrics")
        st.markdown(f"""
        <ul class="data-metrics">
            <li>Total rows before cleaning: <span class="data-metrics-value">{metrics['initial_count']:,}</span></li>
            <li>Total rows after cleaning: <span class="data-metrics-value">{metrics['final_count']:,}</span></li>
            <li>Number of rows dropped: <span class="data-metrics-value">{metrics['rows_dropped']:,}</span></li>
            <li>Number of null values: <span class="data-metrics-value">{metrics['null_values']:,}</span></li>
        </ul>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please ingest the data first.")

with tab4:
    st.header("Model Development and Training")

    # KNN Concept Section
    st.write("""
    ### KNN Concept:
    - K-Nearest Neighbors (KNN) Algorithm: The model is trained using the KNN algorithm, which predicts the outcome based on the 'k' closest data points in the feature space.
    - Distance Metrics: The model uses multiple distance metrics such as Euclidean, Manhattan, and Minkowski to calculate the distance between data points.
    - Weights: Weights can be uniform (all neighbors contribute equally) or distance-based (closer neighbors contribute more to the decision).
    - Grid Search: The model uses GridSearchCV to find the optimal combination of 'k', weights, and distance metrics for the best model performance.
    
    <br> <!-- Adding more space between the bullet points and the image -->
    """, unsafe_allow_html=True)

    # KNN Concept Diagram with adjusted size, centered, and added spacing
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;"> <!-- Adding space between the bullets and image -->
            <img src="https://upload.wikimedia.org/wikipedia/commons/e/e7/KnnClassification.svg" alt="KNN Concept Diagram" style="width: 50%; margin-bottom: 10px;"> <!-- Adjusted size and spacing -->
            <p>KNN Concept Diagram</p>
        </div>
        <br> <!-- Adding space between the caption and the next section -->
        """, unsafe_allow_html=True
    )

    # KNN Decision Boundary Section
    st.write("""
    ### KNN Decision Boundary:
    - The decision boundary created by KNN depends on the value of 'k' and the distribution of the data points in the feature space.
    - As 'k' increases, the decision boundary becomes smoother, which can reduce the model’s sensitivity to noise but might make it less flexible.
    - Different distance metrics can alter the shape and placement of the decision boundary, impacting model performance.
    
    <br> <!-- Adding more space between the bullet points and the image -->
    """, unsafe_allow_html=True)

    # KNN Decision Boundary Image with custom CSS to center and resize, with added spacing
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;"> <!-- Adding space between the bullets and image -->
            <img src="https://miro.medium.com/v2/resize:fit:1400/1*X1B9yhjewtdGBbyENd8KuQ.png" alt="KNN Decision Boundaries" style="width: 60%; margin-bottom: 10px;"> <!-- Adding space between image and caption -->
            <p>KNN Decision Boundaries</p>
        </div>
        <br> <!-- Adding space between the caption and the next section -->
        """, unsafe_allow_html=True
    )

    # KNN Workflow Section
    st.write("""
    ### KNN Workflow:
    - The KNN algorithm involves calculating the distance between the data point of interest and all other points in the dataset.
    - The 'k' nearest neighbors are selected, and the majority class among these neighbors is assigned to the data point.
    - KNN is a lazy learner, meaning it does not build a model until predictions are required, which makes it simple but computationally intensive for large datasets.
    
    <br> <!-- Adding more space between the bullet points and the image -->
    """, unsafe_allow_html=True)

    # KNN Workflow Image with custom CSS to center and resize, with added spacing
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;"> <!-- Adding space between the bullets and image -->
            <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6e8a4917-fbbb-4290-9590-f65640844741_1920x1080.png" alt="KNN Workflow" style="width: 60%; margin-bottom: 10px;"> <!-- Adjusted size and spacing -->
            <p>KNN Workflow</p>
        </div>
        <br> <!-- Adding space between the caption and the next section -->
        """, unsafe_allow_html=True
    )

with tab5:
    st.header("Model Performance Analysis")
    output_path = os.path.join("/Users/iyeng1/Documents/Software-Development/PyDev II/DRIBBLE/OutputLogs/", "final_model_output.csv")
    if os.path.exists(output_path):
        results_df = pd.read_csv(output_path)
        st.write(results_df.head())
    else:
        st.warning("No results found. Please train the model first.")
