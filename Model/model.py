import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate_model(shot_logs_df):
    print("Preparing data for modeling...")
    X = shot_logs_df[['location x', 'location y']].copy()
    Y = shot_logs_df['current shot outcome'].map({'SCORED': 1, 'MISSED': 0}).copy()

    if Y.isnull().any():
        Y.ffill(inplace=True)  # Forward fill to handle NaN values

    random_state = int(time.time())
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    # Define and train various KNN models with different metrics
    metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine', 'chebyshev']
    accuracies = {}
    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, algorithm='auto' if metric != 'cosine' else 'brute')
        knn.fit(X_train, Y_train)
        accuracies[metric] = accuracy_score(Y_test, knn.predict(X_test)) * 100

    # Weighted KNN
    weighted_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    weighted_knn.fit(X_train, Y_train)
    accuracies['weighted'] = accuracy_score(Y_test, weighted_knn.predict(X_test)) * 100

    # Bagging KNN
    bagging_knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_estimators=10, random_state=random_state)
    bagging_knn.fit(X_train, Y_train)
    accuracies['bagging'] = accuracy_score(Y_test, bagging_knn.predict(X_test)) * 100

    # Print accuracies for all techniques
    print("\nModel accuracies for different metrics:")
    for metric, acc in accuracies.items():
        print(f"{metric.capitalize()} Metric: {acc:.2f}%")

    # Create DataFrame for output
    results_df = pd.DataFrame({
        'Player Name': shot_logs_df.loc[X_test.index, 'shoot player'],
        'Location X': X_test['location x'],
        'Location Y': X_test['location y'],
        'Model Prediction': knn.predict(X_test),
        'Actual Outcome': Y_test.values
    })

    return results_df
