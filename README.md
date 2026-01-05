# Is this login weird? 👾

Behavioral login anomaly detection.

## What this is
This project explores **behavior-based login anomaly detection** using simple machine learning models.

Instead of asking *"is this malicious?"*, the system learns what **normal login behavior** looks like and highlights events that don't fit the pattern.

Think: defender intuition, but automated.

This is intentionally small, readable, and security‑oriented.

## Why this exists
Rule-based auth detection breaks fast:

- Attackers adapt
- Rules grow brittle
- Edge cases pile up

Machine learning helps by learning *patterns* instead of hard rules:

- Time-of-day habits
- Location consistency
- Device familiarity
- Login velocity

When something drifts far from "normal", it gets flagged for review.

## What the model looks at
Example features (synthetic or real logs):

- hour\_of\_day
- country
- device\_type
- login\_success
- sessions\_per\_hour

No PII required. No payload inspection. Just behavior.

## How it works (high level)
1. Ingest login events
2. Learn baseline behavior
3. Score new logins by how unusual they are
4. Surface the weird stuff

Models explored:

- Isolation Forest

## What this is *not*
- ❌ A production auth system
- ❌ A silver bullet detector
- ❌ A giant deep learning model

This is a **thinking tool** for blue teamers.

## Example question this answers
"Would *I* trust this login?"

If the model hesitates, you probably should too.

## Future ideas
- k-means clustering
- simple statistical baselines
- Visualize "normal" vs "weird"

## Disclaimer
This project uses synthetic or anonymized data only. No real credentials, no real users.

<br>
