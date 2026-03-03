"""Matplotlib charts for anomaly analysis."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("login_anomaly")


def plot_score_distribution(ax, test_df):
    """Histogram of anomaly scores, split by true label."""
    normal = test_df[test_df["true_anomaly"] == 0]["anomaly_score"].dropna()
    anomalous = test_df[test_df["true_anomaly"] == 1]["anomaly_score"].dropna()

    ax.hist(normal, bins=30, alpha=0.7, label="Normal", color="green")
    ax.hist(anomalous, bins=30, alpha=0.7, label="True Anomaly", color="red")
    ax.axvline(x=-0.7, color="black", linestyle="--", label="Potential Threshold")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()


def plot_false_positives_by_hour(ax, false_positives, true_negatives):
    """Side-by-side bar chart: normal vs false-positive counts per hour."""
    if false_positives.empty:
        ax.set_title("False Positives by Hour (none)")
        return

    fp_hours = false_positives["hour_of_day"].value_counts().sort_index()
    normal_hours = true_negatives["hour_of_day"].value_counts().sort_index()

    hours_range = range(24)
    fp_counts = [fp_hours.get(h, 0) for h in hours_range]
    normal_counts = [normal_hours.get(h, 0) for h in hours_range]

    x_positions = np.arange(len(hours_range))
    width = 0.35

    ax.bar(x_positions - width / 2, normal_counts, width,
           label="Normal", alpha=0.7, color="green")
    ax.bar(x_positions + width / 2, fp_counts, width,
           label="False Positive", alpha=0.7, color="orange")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")
    ax.set_title("False Positives by Hour")
    ax.legend()
    ax.set_xticks(x_positions[::2])
    ax.set_xticklabels(list(hours_range)[::2])


def plot_false_positives_by_country(ax, false_positives, true_negatives):
    """Side-by-side bar chart: normal vs false-positive counts per country."""
    if false_positives.empty:
        ax.set_title("False Positives by Country (none)")
        return

    fp_countries = false_positives["country"].value_counts()
    normal_countries = true_negatives["country"].value_counts()

    all_countries = sorted(set(fp_countries.index) | set(normal_countries.index))
    fp_counts = [fp_countries.get(c, 0) for c in all_countries]
    normal_counts = [normal_countries.get(c, 0) for c in all_countries]

    x_positions = np.arange(len(all_countries))
    width = 0.35

    ax.bar(x_positions - width / 2, normal_counts, width,
           label="Normal", alpha=0.7, color="green")
    ax.bar(x_positions + width / 2, fp_counts, width,
           label="False Positive", alpha=0.7, color="orange")
    ax.set_xlabel("Country")
    ax.set_ylabel("Count")
    ax.set_title("False Positives by Country")
    ax.legend()
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_countries)


def save_analysis_plots(test_df, false_positives, true_negatives, path, dpi):
    """Render all three analysis plots and save to disk."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    plot_score_distribution(axes[0], test_df)
    plot_false_positives_by_hour(axes[1], false_positives, true_negatives)
    plot_false_positives_by_country(axes[2], false_positives, true_negatives)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Visualization saved to %s", path)
