"""Model evaluation: confusion matrix, classification report, FP/TP analysis."""

import logging

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger("login_anomaly")

DISPLAY_COLUMNS = [
    "hour_of_day",
    "country",
    "device_type",
    "login_success",
    "sessions_per_hour",
    "anomaly_score",
]


def log_confusion_matrix(y_true, y_pred):
    """Log the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion matrix:\n%s", cm)
    return cm


def log_classification_report(y_true, y_pred):
    """Log sklearn's classification report."""
    report = classification_report(y_true, y_pred)
    logger.info("Classification report:\n%s", report)
    return report


def extract_subsets(test_df):
    """Split test results into false positives, true positives, true negatives."""
    false_positives = test_df[
        (test_df["predicted_anomaly"] == 1) & (test_df["true_anomaly"] == 0)
    ]
    true_positives = test_df[
        (test_df["predicted_anomaly"] == 1) & (test_df["true_anomaly"] == 1)
    ]
    true_negatives = test_df[
        (test_df["predicted_anomaly"] == 0) & (test_df["true_anomaly"] == 0)
    ]
    return false_positives, true_positives, true_negatives


def log_subset_details(label, subset_df):
    """Log a readable summary of a prediction subset."""
    cols = [c for c in DISPLAY_COLUMNS if c in subset_df.columns]
    logger.info("%s (%d cases):\n%s", label, len(subset_df), subset_df[cols])
