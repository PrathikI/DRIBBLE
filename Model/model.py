import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate_model(shot_logs_df):
    print("Preparing data for modeling...")
    
    X = shot_logs_df[['LOC_X', 'LOC_Y']]  
    Y = shot_logs_df['SHOT_MADE'].astype(int)  

    if Y.isnull().any():
        Y.ffill(inplace=True)  # Handling NaNs by forward filling

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Setup the grid of parameters to test
    param_grid = {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
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
    test_accuracy = accuracy_score(Y_test, Y_pred) * 100  
    print(f"Test set accuracy: {test_accuracy:.2f}%")

    # Debugging: Print samples to verify correct data alignment
    print("Sample of X_test data:")
    print(X_test.head())

    print("Sample of corresponding Player Names:")
    print(shot_logs_df.loc[X_test.index, 'PLAYER_NAME'].head())

    print("Sample of Model Predictions vs Actual Outcomes:")
    print(pd.DataFrame({'Prediction': Y_pred, 'Actual': Y_test}).head())

    # Prepare results DataFrame with proper alignment
    results_df = pd.DataFrame({
        'Player Name': shot_logs_df.loc[X_test.index, 'PLAYER_NAME'],  
        'Location X': X_test['LOC_X'].values,  
        'Location Y': X_test['LOC_Y'].values,
        'Model Prediction': Y_pred,
        'Actual Outcome': Y_test.values  
    })

    return results_df
