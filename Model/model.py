from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_and_evaluate_model(shot_logs_df):
    print("Preparing data for modeling...")
    
    # Define the features used for the model
    features = [
        'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'QUARTER', 'SHOT_TYPE',
        'MINS_LEFT', 'SECS_LEFT', 'BASIC_ZONE', 'ZONE_NAME'
    ]
    X = shot_logs_df[features]

    # One-Hot Encode categorical variables
    X = pd.get_dummies(X, columns=['SHOT_TYPE', 'BASIC_ZONE', 'ZONE_NAME'])

    Y = shot_logs_df['SHOT_MADE'].astype(int)

    if Y.isnull().any():
        Y.ffill(inplace=True)

    # Sample the data to reduce size for faster processing
    X_sample, Y_sample = X.sample(frac=0.2, random_state=42), Y.sample(frac=0.2, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_sample, Y_sample, test_size=0.2, random_state=42)

    # Setup the grid of parameters to test
    param_dist = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Create a KNeighborsClassifier and RandomizedSearchCV object
    knn = KNeighborsClassifier()
    random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X_train, Y_train)

    # Evaluate the best model on the test set
    best_knn = random_search.best_estimator_
    Y_pred = best_knn.predict(X_test)

    # Prepare results DataFrame with proper alignment
    results_df = pd.DataFrame({
        'Player Name': shot_logs_df.loc[X_test.index, 'PLAYER_NAME'],
        'Location X': X_test['LOC_X'],
        'Location Y': X_test['LOC_Y'],
        'Shot Distance': X_test['SHOT_DISTANCE'],
        'Quarter': X_test['QUARTER'],
        'Minutes Left': X_test['MINS_LEFT'],
        'Seconds Left': X_test['SECS_LEFT'],
        'Basic Zone': shot_logs_df.loc[X_test.index, 'BASIC_ZONE'],
        'Zone Name': shot_logs_df.loc[X_test.index, 'ZONE_NAME'],
        'Model Prediction': Y_pred,
        'Actual Outcome': Y_test
    })

    return results_df
