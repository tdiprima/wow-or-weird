"""Pipeline configuration loaded from environment variables with sane defaults."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the anomaly detection pipeline."""

    # Dataset
    n_samples: int = 5000
    random_seed: int = 42
    test_size: float = 0.2

    # Model
    contamination: float = 0.005
    n_estimators: int = 100

    # Alerting
    slack_webhook_url: str = ""
    alert_score_threshold: float = -0.72

    # Output
    visualization_path: str = "anomaly_analysis.png"
    visualization_dpi: int = 150


def load_config() -> PipelineConfig:
    """Build config from environment variables, falling back to defaults."""
    return PipelineConfig(
        n_samples=int(os.getenv("AD_N_SAMPLES", "5000")),
        random_seed=int(os.getenv("AD_RANDOM_SEED", "42")),
        test_size=float(os.getenv("AD_TEST_SIZE", "0.2")),
        contamination=float(os.getenv("AD_CONTAMINATION", "0.005")),
        n_estimators=int(os.getenv("AD_N_ESTIMATORS", "100")),
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
        alert_score_threshold=float(os.getenv("AD_ALERT_THRESHOLD", "-0.72")),
        visualization_path=os.getenv("AD_VIZ_PATH", "anomaly_analysis.png"),
        visualization_dpi=int(os.getenv("AD_VIZ_DPI", "150")),
    )


def validate_config(config: PipelineConfig) -> None:
    """Raise ValueError if any config value is nonsensical."""
    if config.n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {config.n_samples}")
    if not 0 < config.test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {config.test_size}")
    if not 0 < config.contamination < 0.5:
        raise ValueError(
            f"contamination must be in (0, 0.5), got {config.contamination}"
        )
    if config.n_estimators < 1:
        raise ValueError(
            f"n_estimators must be positive, got {config.n_estimators}"
        )
