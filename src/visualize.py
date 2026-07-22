"""
visualize.py
------------
Generates the core visualizations for the A/B test analysis report:
funnel comparison, conversion rate with CI, and segment breakdown.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.stats_tests import two_proportion_ztest, segment_analysis, funnel_dropoff

DATA_PATH = Path(__file__).parent.parent / "data" / "ab_test_data.csv"
OUT_DIR = Path(__file__).parent.parent / "outputs"


def plot_funnel(df: pd.DataFrame):
    funnel = funnel_dropoff(df)
    stages = ["viewed_product_rate_from_start", "added_to_cart_rate_from_start", "purchased_rate_from_start"]
    labels = ["Viewed", "Added to Cart", "Purchased"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(stages))
    width = 0.35
    for i, group in enumerate(["control", "treatment"]):
        vals = funnel[funnel["group"] == group][stages].values.flatten()
        offset = -width / 2 if group == "control" else width / 2
        ax.bar([xi + offset for xi in x], vals, width, label=group)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Conversion rate from start")
    ax.set_title("Funnel Conversion by Group")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "funnel_comparison.png", dpi=120)
    plt.close()
    print("Saved outputs/funnel_comparison.png")


def plot_conversion_with_ci(df: pd.DataFrame):
    result = two_proportion_ztest(df, "purchased")
    groups = ["control", "treatment"]
    rates = [result["rate_control"], result["rate_treatment"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(groups, rates, color=["#94a3b8", "#2563eb"])
    ax.set_ylabel("Purchase conversion rate")
    ax.set_title(f"Conversion Rate by Group\n(relative lift: {result['relative_lift_pct']}%, p={result['p_value']})")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate, f"{rate:.4f}",
                ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "conversion_rate.png", dpi=120)
    plt.close()
    print("Saved outputs/conversion_rate.png")


def plot_segment_breakdown(df: pd.DataFrame):
    seg = segment_analysis(df, "device", "purchased")
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(seg))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], seg["rate_control"], width, label="control", color="#94a3b8")
    ax.bar([xi + width / 2 for xi in x], seg["rate_treatment"], width, label="treatment", color="#2563eb")
    ax.set_xticks(list(x))
    ax.set_xticklabels(seg["device"])
    ax.set_ylabel("Purchase conversion rate")
    ax.set_title("Conversion Rate by Device Segment")
    for i, row in seg.iterrows():
        sig = "significant" if row["significant_at_05"] else "not significant"
        ax.text(i, max(row["rate_control"], row["rate_treatment"]) + 0.0005,
                f"{row['relative_lift_pct']}%\n({sig})", ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "segment_breakdown.png", dpi=120)
    plt.close()
    print("Saved outputs/segment_breakdown.png")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    plot_funnel(df)
    plot_conversion_with_ci(df)
    plot_segment_breakdown(df)
