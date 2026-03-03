import numpy as np
import pandas as pd

np.random.seed(42)

# Generate normal patterns
n_samples = 5000

# Normal patterns with realistic distributions
hours = np.random.choice(
    range(24),
    n_samples,
    p=[
        0.02,
        0.01,
        0.01,
        0.01,
        0.01,
        0.02,  # 0-5am (low)
        0.03,
        0.04,
        0.06,
        0.08,
        0.06,
        0.05,  # 6-11am (rising)
        0.06,
        0.05,
        0.04,
        0.04,
        0.05,
        0.06,  # 12-5pm (steady)
        0.08,
        0.07,
        0.06,
        0.04,
        0.03,
        0.02,  # 6-11pm (declining)
    ],
)

countries = np.random.choice(
    ["US", "UK", "CA", "DE", "FR", "AU"], n_samples, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
)

devices = np.random.choice(
    ["mobile", "desktop", "tablet"], n_samples, p=[0.6, 0.35, 0.05]
)

# Most logins succeed
login_success = np.random.choice([True, False], n_samples, p=[0.95, 0.05])

# Sessions per hour varies by device
sessions = np.where(
    devices == "mobile",
    np.random.poisson(3, n_samples),
    np.where(
        devices == "desktop",
        np.random.poisson(5, n_samples),
        np.random.poisson(2, n_samples),
    ),
)

df = pd.DataFrame(
    {
        "hour_of_day": hours,
        "country": countries,
        "device_type": devices,
        "login_success": login_success,
        "sessions_per_hour": sessions,
    }
)

# Add some anomalies (optional, for testing)
anomalies = pd.DataFrame(
    {
        "hour_of_day": [3, 2, 22],
        "country": ["RU", "CN", "BR"],  # Unusual countries
        "device_type": ["desktop", "desktop", "mobile"],
        "login_success": [False, False, True],
        "sessions_per_hour": [15, 20, 25],  # Unusually high
    }
)

df = pd.concat([df, anomalies], ignore_index=True)

df.to_csv("logins_raw.csv")

print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"\nValue counts:\n{df['country'].value_counts()}")
