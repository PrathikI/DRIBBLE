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

    # Handle NaN values using forward fill
    if Y.isnull().any():
        Y.ffill(inplace=True)  # Proper method to forward fill

    random_state = int(time.time())
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    # Evaluate different distance metrics
    distance_metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine', 'chebyshev']
    accuracies = {}

    for metric in distance_metrics:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, algorithm='auto' if metric != 'cosine' else 'brute')
        knn.fit(X_train, Y_train)
        accuracies[metric] = accuracy_score(Y_test, knn.predict(X_test)) * 100

    # Weighted KNN
    weighted_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    weighted_knn.fit(X_train, Y_train)
    accuracies['weighted'] = accuracy_score(Y_test, weighted_knn.predict(X_test)) * 100

    # Bagging KNN
    bagging_knn = BaggingClassifier(
        estimator=KNeighborsClassifier(n_neighbors=5),
        n_estimators=10,
        random_state=42,
        max_samples=0.5,
        max_features=0.5
    )
    bagging_knn.fit(X_train, Y_train)
    accuracies['bagging'] = accuracy_score(Y_test, bagging_knn.predict(X_test)) * 100

    return accuracies
