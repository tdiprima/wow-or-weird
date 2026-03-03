# Wow or Weird?

*"Wow or weird?" — Wagstaff, Bob's Burgers*

A behavior-based login anomaly detection experiment using **Isolation Forest** and scikit-learn.

## The Problem

Security teams drown in login events. Most are normal. A few are sketchy. The challenge: flag the sketchy ones without crying wolf on legitimate users logging in from a coffee shop at 11pm.

This project explores unsupervised anomaly detection on login data — no labeled attack data required. The model learns what "normal" looks like and surfaces the outliers.

## What It Does

- Generates synthetic login data (5,000+ events) with realistic behavioral patterns
- Injects known anomalies: off-hours logins, unusual countries, excessive session counts
- Trains an Isolation Forest to score each login's weirdness
- Explains *why* a login looks suspicious (feature contribution analysis)
- Sends a Slack alert when a high-severity anomaly is detected

**Features used:** hour of day, country, device type, login success, sessions per hour

## Quick Start

```bash
# Install dependencies
uv sync

# (Optional) Generate standalone CSV data
python scripts/generate_synthetic_data.py

# Run the full detection pipeline
python src/login_anomaly_detection.py
```

To enable Slack alerts, set your webhook URL:

```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

Output includes a confusion matrix, classification report, false positive breakdown, and `anomaly_analysis.png`.

## Stack

Python · pandas · scikit-learn · matplotlib · Slack Webhooks

## License

[MIT](LICENSE)

<br>
