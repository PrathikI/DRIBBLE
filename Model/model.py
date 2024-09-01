import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate_model(shot_logs_df):
    print("Preparing data for modeling...")
    X = shot_logs_df[['location x', 'location y']].copy()
    Y = shot_logs_df['current shot outcome'].map({'SCORED': 1, 'MISSED': 0}).copy()

    if Y.isnull().any():
        Y.ffill(inplace=True) 

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Setup the grid of parameters to test
    param_grid = {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance']
    }

    # Create a KNeighborsClassifier and GridSearchCV object
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)

    # Best parameters and best score
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_ * 100 
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validated score: {best_cv_score:.2f}%")

    # Evaluate the best model on the test set
    best_knn = grid_search.best_estimator_
    Y_pred = best_knn.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred) * 100  # Convert to percentage
    print(f"Test set accuracy: {test_accuracy:.2f}%")

    # Prepare results DataFrame with additional details
    results_df = pd.DataFrame({
        'Player Name': shot_logs_df.loc[X_test.index, 'shoot player'],
        'Location X': X_test['location x'],
        'Location Y': X_test['location y'],
        'Model Prediction': Y_pred,
        'Actual Outcome': Y_test.values
    })

    return results_df
