"""Login Anomaly Detection Pipeline - orchestration entry point."""

import pandas as pd

from alerting import build_alert_payload, determine_severity, send_slack_alert
from config import load_config, validate_config
from data_generator import build_dataset
from encoding import FEATURE_COLUMNS, encode_features, encode_new_login
from evaluation import (
    extract_subsets,
    log_classification_report,
    log_confusion_matrix,
    log_subset_details,
)
from logging_config import configure_logging
from model import (
    compute_feature_contributions,
    predict,
    rank_contributions,
    split_data,
    train_isolation_forest,
)
from visualization import save_analysis_plots


def log_feature_contributions(logger, label, row_dict, score, contributions):
    """Pretty-print feature contribution analysis for a single row."""
    logger.info("--- %s ---", label)
    logger.info("Row: %s", row_dict)
    logger.info("Anomaly score: %.4f", score)
    logger.info("Feature contributions (higher = more suspicious):")
    for feat, contrib in rank_contributions(contributions):
        logger.info("  %s: %.4f", feat, contrib)


def run_pipeline():
    """Execute the full anomaly detection pipeline."""
    logger = configure_logging()
    config = load_config()
    validate_config(config)

    # -- Data --
    df = build_dataset(config.n_samples, config.random_seed)
    logger.info(
        "Dataset: %d rows, %d true anomalies",
        len(df),
        df["true_anomaly"].sum(),
    )

    # -- Encoding --
    df_encoded, encoders = encode_features(df)
    feature_matrix = df_encoded[FEATURE_COLUMNS]

    # -- Train / Test --
    X_train, X_test, y_train, y_test = split_data(
        feature_matrix, df["true_anomaly"], config.test_size, config.random_seed
    )

    # -- Model --
    model = train_isolation_forest(
        X_train, config.contamination, config.n_estimators, config.random_seed
    )

    predictions, scores = predict(model, X_test)

    # Write results back for analysis
    df.loc[X_test.index, "predicted_anomaly"] = predictions
    df.loc[X_test.index, "anomaly_score"] = scores

    # -- Evaluation --
    log_confusion_matrix(y_test, predictions)
    log_classification_report(y_test, predictions)

    test_df = df.loc[X_test.index].copy()
    false_positives, true_positives, true_negatives = extract_subsets(test_df)

    log_subset_details("FALSE POSITIVES", false_positives)
    log_subset_details("TRUE POSITIVES", true_positives)

    # -- Feature contributions --
    display_cols = [
        "hour_of_day", "country", "device_type",
        "login_success", "sessions_per_hour",
    ]

    if not false_positives.empty:
        fp_idx = false_positives.index[0]
        fp_contributions = compute_feature_contributions(
            feature_matrix.loc[fp_idx].values, model, FEATURE_COLUMNS, X_train
        )
        log_feature_contributions(
            logger,
            "FALSE POSITIVE EXAMPLE",
            df.loc[fp_idx, display_cols].to_dict(),
            df.loc[fp_idx, "anomaly_score"],
            fp_contributions,
        )

    if not true_positives.empty:
        tp_idx = true_positives.index[0]
        tp_contributions = compute_feature_contributions(
            feature_matrix.loc[tp_idx].values, model, FEATURE_COLUMNS, X_train
        )
        log_feature_contributions(
            logger,
            "TRUE POSITIVE EXAMPLE",
            df.loc[tp_idx, display_cols].to_dict(),
            df.loc[tp_idx, "anomaly_score"],
            tp_contributions,
        )

    # -- Visualization --
    save_analysis_plots(
        test_df, false_positives, true_negatives,
        config.visualization_path, config.visualization_dpi,
    )

    # -- New login test --
    new_login = pd.DataFrame({
        "hour_of_day": [3],
        "country": ["RU"],
        "device_type": ["desktop"],
        "login_success": [False],
        "sessions_per_hour": [15],
    })

    new_login_encoded = encode_new_login(new_login, encoders)
    new_X = new_login_encoded[FEATURE_COLUMNS]

    new_pred, new_score = predict(model, new_X)
    is_anomaly = bool(new_pred[0])

    logger.info("New login: %s", new_login.iloc[0].to_dict())
    logger.info("Anomaly score: %.4f", new_score[0])
    logger.info("Result: %s", "ANOMALY DETECTED" if is_anomaly else "Normal login")

    new_contributions = compute_feature_contributions(
        new_X.values[0], model, FEATURE_COLUMNS, X_train
    )
    ranked = rank_contributions(new_contributions)
    for feat, contrib in ranked:
        logger.info("  %s: %.4f", feat, contrib)

    # -- Slack alert --
    if is_anomaly and new_score[0] < config.alert_score_threshold:
        severity = determine_severity(new_score[0])
        payload = build_alert_payload(
            new_login.iloc[0], new_score[0], severity, ranked[:3]
        )
        try:
            send_slack_alert(config.slack_webhook_url, payload)
        except RuntimeError as exc:
            logger.warning("Failed to send Slack alert: %s", exc)
    else:
        logger.info("No alert triggered (not anomalous or score above threshold)")


if __name__ == "__main__":
    run_pipeline()
