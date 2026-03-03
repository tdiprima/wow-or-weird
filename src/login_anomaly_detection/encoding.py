"""Categorical feature encoding for the login dataset."""

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "hour_of_day",
    "country_encoded",
    "device_encoded",
    "login_success_encoded",
    "sessions_per_hour",
]


@dataclass
class Encoders:
    """Container for fitted label encoders so they can be reused at inference."""

    country: LabelEncoder
    device: LabelEncoder
    login_success: LabelEncoder


def encode_features(df):
    """Label-encode categorical columns. Returns (enriched_df, Encoders)."""
    encoded = df.copy()

    le_country = LabelEncoder()
    le_device = LabelEncoder()
    le_success = LabelEncoder()

    encoded["country_encoded"] = le_country.fit_transform(df["country"])
    encoded["device_encoded"] = le_device.fit_transform(df["device_type"])
    encoded["login_success_encoded"] = le_success.fit_transform(df["login_success"])

    encoders = Encoders(country=le_country, device=le_device, login_success=le_success)
    return encoded, encoders


def encode_new_login(login, encoders):
    """Encode a new login event using previously fitted encoders."""
    login = login.copy()
    login["country_encoded"] = encoders.country.transform(login["country"])
    login["device_encoded"] = encoders.device.transform(login["device_type"])
    login["login_success_encoded"] = encoders.login_success.transform(login["login_success"])
    return login
