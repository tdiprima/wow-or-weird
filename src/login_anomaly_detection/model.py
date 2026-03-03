"""Isolation Forest training, prediction, and feature contribution analysis."""

import operator

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


def split_data(feature_matrix, labels, test_size, random_seed):
    """Stratified train/test split. Returns (X_train, X_test, y_train, y_test)."""
    return train_test_split(
        feature_matrix,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )


def train_isolation_forest(X_train, contamination, n_estimators, random_seed):
    """Fit an Isolation Forest on the training data and return the model."""
    model = IsolationForest(
        contamination=contamination,
        random_state=random_seed,
        n_estimators=n_estimators,
    )
    model.fit(X_train)
    return model


def predict(model, X_test):
    """Run predictions and return (binary_labels, anomaly_scores).

    binary_labels: 1 = anomaly, 0 = normal
    anomaly_scores: lower = more suspicious
    """
    raw_predictions = model.predict(X_test)
    scores = model.score_samples(X_test)
    binary_labels = (raw_predictions == -1).astype(int)
    return binary_labels, scores


def compute_feature_contributions(row_values, model, feature_names, X_train):
    """Measure each feature's contribution to an anomaly score.

    For each feature, replace it with the training median and re-score.
    A large positive delta means that feature was driving the anomaly.
    """
    row_df = pd.DataFrame([row_values], columns=feature_names)
    base_score = model.score_samples(row_df)[0]

    contributions = {}
    for idx, feature in enumerate(feature_names):
        modified = row_df.copy()
        modified.iloc[0, idx] = X_train[feature].median()
        new_score = model.score_samples(modified)[0]
        contributions[feature] = new_score - base_score

    return contributions


def rank_contributions(contributions):
    """Return contributions sorted most-suspicious-first."""
    return sorted(contributions.items(), key=operator.itemgetter(1), reverse=True)
