"""Slack webhook alerting for detected anomalies."""

import json
import logging

import requests

logger = logging.getLogger("login_anomaly")


def build_alert_payload(login_row, anomaly_score, severity, top_features):
    """Construct a Slack Block Kit payload for an anomalous login."""
    feature_text = "\n".join(
        [f"  *{feat}*: {contrib:.4f}" for feat, contrib in top_features]
    )

    return {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Anomalous Login Detected",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"},
                    {"type": "mrkdwn", "text": f"*Anomaly Score:*\n{anomaly_score:.4f}"},
                    {"type": "mrkdwn", "text": f"*Hour of Day:*\n{login_row['hour_of_day']}"},
                    {"type": "mrkdwn", "text": f"*Country:*\n{login_row['country']}"},
                    {"type": "mrkdwn", "text": f"*Device:*\n{login_row['device_type']}"},
                    {"type": "mrkdwn", "text": f"*Sessions/hr:*\n{login_row['sessions_per_hour']}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Top Suspicious Features:*\n" + feature_text,
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Model: Isolation Forest | Detection Type: Behavioral Anomaly",
                    }
                ],
            },
        ]
    }


def send_slack_alert(webhook_url, payload):
    """POST the payload to a Slack incoming webhook.

    Raises RuntimeError on failure so callers can decide how to handle it.
    """
    if not webhook_url:
        raise RuntimeError("No SLACK_WEBHOOK_URL configured")

    response = requests.post(
        webhook_url,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Slack webhook returned {response.status_code}: {response.text}"
        )

    logger.info("Slack alert sent successfully")


def determine_severity(anomaly_score):
    """Map a raw anomaly score to a human-readable severity label."""
    if anomaly_score < -0.8:
        return "HIGH"
    return "MEDIUM"
