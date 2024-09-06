from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import random

def train_and_evaluate_model(shot_logs_df):
    print("Preparing data for modeling...")
    
    # Define the features used for the model
    features = [
        'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'QUARTER', 'SHOT_TYPE',
        'MINS_LEFT', 'SECS_LEFT', 'BASIC_ZONE', 'ZONE_NAME'
    ]
    X = pd.get_dummies(shot_logs_df[features], columns=['SHOT_TYPE', 'BASIC_ZONE', 'ZONE_NAME'])
    Y = shot_logs_df['SHOT_MADE'].astype(int).fillna(method='ffill')

    # Randomize the data split for variability
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    # Setup the grid of parameters to test, expanding the search
    # Accuracies a lot better with these parameters; takes a while and makes laptop process harder
    param_dist = {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Create a KNeighborsClassifier and RandomizedSearchCV object
    knn = KNeighborsClassifier()
    
    # Randomize the seed for the hyperparameter search to introduce variability
    random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=random.randint(0, 10000))
    random_search.fit(X_train, Y_train)

    # Evaluate the best model on the test set
    best_knn = random_search.best_estimator_
    Y_pred = best_knn.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)

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

    return results_df, best_knn, X_test, Y_test, random_search.best_params_, random_search.best_score_ * 100, test_accuracy
