"""Synthetic login dataset generation."""

import numpy as np
import pandas as pd

# fmt: off
HOUR_PROBABILITIES = {
    # Late night / early morning — very low activity
    0: 0.02,  1: 0.01,  2: 0.01,  3: 0.01,  4: 0.01,  5: 0.02,
    # Morning ramp-up
    6: 0.03,  7: 0.04,  8: 0.06,  9: 0.08, 10: 0.06, 11: 0.05,
    # Afternoon plateau
   12: 0.06, 13: 0.05, 14: 0.04, 15: 0.04, 16: 0.05, 17: 0.06,
    # Evening wind-down
   18: 0.08, 19: 0.07, 20: 0.06, 21: 0.04, 22: 0.03, 23: 0.02,
}
# fmt: on

COUNTRY_WEIGHTS = {"US": 0.4, "UK": 0.2, "CA": 0.15, "DE": 0.1, "FR": 0.1, "AU": 0.05}
DEVICE_WEIGHTS = {"mobile": 0.6, "desktop": 0.35, "tablet": 0.05}

KNOWN_ANOMALIES = [
    {"hour_of_day": 3,  "country": "RU", "device_type": "desktop", "login_success": False, "sessions_per_hour": 15},
    {"hour_of_day": 2,  "country": "CN", "device_type": "desktop", "login_success": False, "sessions_per_hour": 20},
    {"hour_of_day": 22, "country": "BR", "device_type": "mobile",  "login_success": True,  "sessions_per_hour": 25},
]


def generate_normal_logins(n_samples: int, seed: int) -> pd.DataFrame:
    """Generate synthetic normal login events."""
    rng = np.random.RandomState(seed)

    hours = list(HOUR_PROBABILITIES.keys())
    hour_probs = list(HOUR_PROBABILITIES.values())

    countries = list(COUNTRY_WEIGHTS.keys())
    country_probs = list(COUNTRY_WEIGHTS.values())

    devices = list(DEVICE_WEIGHTS.keys())
    device_probs = list(DEVICE_WEIGHTS.values())

    hour_col = rng.choice(hours, n_samples, p=hour_probs)
    country_col = rng.choice(countries, n_samples, p=country_probs)
    device_col = rng.choice(devices, n_samples, p=device_probs)
    success_col = rng.choice([True, False], n_samples, p=[0.95, 0.05])

    sessions_col = np.where(
        device_col == "mobile",
        rng.poisson(3, n_samples),
        np.where(
            device_col == "desktop",
            rng.poisson(5, n_samples),
            rng.poisson(2, n_samples),
        ),
    )

    return pd.DataFrame({
        "hour_of_day": hour_col,
        "country": country_col,
        "device_type": device_col,
        "login_success": success_col,
        "sessions_per_hour": sessions_col,
    })


def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Append known anomalous rows and label all rows (0=normal, 1=anomaly)."""
    anomalies = pd.DataFrame(KNOWN_ANOMALIES)
    combined = pd.concat([df, anomalies], ignore_index=True)

    combined["true_anomaly"] = 0
    combined.loc[combined.index[-len(KNOWN_ANOMALIES):], "true_anomaly"] = 1

    return combined


def build_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    """Create the full labelled dataset: normal logins + injected anomalies."""
    normal = generate_normal_logins(n_samples, seed)
    return inject_anomalies(normal)
