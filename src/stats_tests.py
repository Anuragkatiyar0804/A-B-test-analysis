"""
stats_tests.py
--------------
Core statistical toolkit for analyzing the A/B test: SRM check, power
analysis, two-proportion z-test, t-test for revenue, and segment-level
(heterogeneous effect) analysis.

Each function is written to be explained standalone in an interview —
that's the point of separating them rather than writing one big script.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest


def check_sample_ratio_mismatch(df: pd.DataFrame, expected_ratio: float = 0.5) -> dict:
    """
    Sample Ratio Mismatch (SRM) check via chi-square goodness-of-fit test.

    Why this must run FIRST, before any effect analysis:
    If your randomization/logging pipeline is broken (e.g. a bug drops
    treatment-group events disproportionately, as we simulated), your
    group sizes won't match the intended split. Any effect you "detect"
    downstream could be an artifact of WHO ended up in each group, not a
    true treatment effect. A significant SRM p-value (typically p < 0.01,
    NOT 0.05 — SRM checks use a stricter threshold since a false negative
    here invalidates the whole experiment) means: stop, do not trust the
    experiment's results, go find the pipeline bug first.
    """
    counts = df["group"].value_counts()
    n_control = counts.get("control", 0)
    n_treatment = counts.get("treatment", 0)
    total = n_control + n_treatment

    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)

    chi2, p_value = stats.chisquare(
        [n_control, n_treatment],
        f_exp=[expected_control, expected_treatment]
    )

    return {
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "observed_ratio": round(n_control / total, 4),
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "srm_detected": p_value < 0.01,
    }


def power_analysis(baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> dict:
    """
    Minimum sample size needed per group to reliably detect a given
    Minimum Detectable Effect (MDE), BEFORE running the experiment.

    Why this belongs in the project even though we already have the data:
    In a real job, this is run BEFORE launching a test, to decide how long
    it needs to run. Showing you know to do this — not just analyze data
    after the fact — is exactly what separates "I ran a t-test" from
    "I understand experimental design."
    """
    effect_size = (
        2 * np.arcsin(np.sqrt(baseline_rate + mde)) - 2 * np.arcsin(np.sqrt(baseline_rate))
    )
    analysis = NormalIndPower()
    n_per_group = analysis.solve_power(
        effect_size=abs(effect_size), alpha=alpha, power=power, ratio=1.0
    )
    return {
        "baseline_rate": baseline_rate,
        "mde": mde,
        "alpha": alpha,
        "power": power,
        "required_n_per_group": int(np.ceil(n_per_group)),
    }


def two_proportion_ztest(df: pd.DataFrame, metric_col: str = "purchased") -> dict:
    """
    Two-proportion z-test comparing conversion rate between groups.

    Why z-test, not chi-square, for this: with a directional hypothesis
    (treatment should INCREASE conversion, not just differ), a one-sided
    z-test is more appropriate and slightly more powerful than a two-sided
    chi-square test of independence, though both are defensible — be ready
    to say why you picked one.
    """
    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    successes = np.array([treatment[metric_col].sum(), control[metric_col].sum()])
    nobs = np.array([len(treatment), len(control)])

    z_stat, p_value = proportions_ztest(successes, nobs, alternative="larger")

    rate_control = control[metric_col].mean()
    rate_treatment = treatment[metric_col].mean()
    relative_lift = (rate_treatment - rate_control) / rate_control

    # 95% CI on the difference in proportions
    p1, p2 = rate_treatment, rate_control
    n1, n2 = len(treatment), len(control)
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "metric": metric_col,
        "rate_control": round(rate_control, 5),
        "rate_treatment": round(rate_treatment, 5),
        "absolute_diff": round(diff, 5),
        "relative_lift_pct": round(relative_lift * 100, 2),
        "ci_95_diff": (round(ci_low, 5), round(ci_high, 5)),
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": p_value < 0.05,
    }


def revenue_ttest(df: pd.DataFrame) -> dict:
    """
    Welch's t-test (unequal variance) comparing revenue per purchasing
    user between groups. Welch's, not Student's, because we should not
    assume equal variances between groups without checking — Welch's is
    the safer default and costs almost nothing in power when variances
    ARE equal.
    """
    control_rev = df[(df["group"] == "control") & (df["purchased"])]["revenue"]
    treatment_rev = df[(df["group"] == "treatment") & (df["purchased"])]["revenue"]

    t_stat, p_value = stats.ttest_ind(treatment_rev, control_rev, equal_var=False)

    return {
        "mean_revenue_control": round(control_rev.mean(), 2),
        "mean_revenue_treatment": round(treatment_rev.mean(), 2),
        "diff": round(treatment_rev.mean() - control_rev.mean(), 2),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": p_value < 0.05,
        "n_control": len(control_rev),
        "n_treatment": len(treatment_rev),
    }


def segment_analysis(df: pd.DataFrame, segment_col: str = "device", metric_col: str = "purchased") -> pd.DataFrame:
    """
    Breaks down the treatment effect by segment (e.g. device type).

    Why this is not optional: a pooled/overall result can hide the fact
    that an effect is concentrated in (or entirely driven by) one segment
    — a mild form of Simpson's paradox risk. A real product decision
    ("should we ship this to everyone?") often depends on this breakdown,
    e.g. if mobile shows strong lift but desktop shows none or slightly
    negative, "ship to mobile only" is a materially different, better
    recommendation than "ship to everyone."
    """
    results = []
    for segment_value in df[segment_col].unique():
        seg_df = df[df[segment_col] == segment_value]
        r = two_proportion_ztest(seg_df, metric_col)
        r[segment_col] = segment_value
        results.append(r)
    return pd.DataFrame(results)[[segment_col, "rate_control", "rate_treatment",
                                    "relative_lift_pct", "p_value", "significant_at_05"]]


def funnel_dropoff(df: pd.DataFrame) -> pd.DataFrame:
    """Computes stage-wise funnel conversion by group."""
    stages = ["viewed_product", "added_to_cart", "purchased"]
    rows = []
    for group in ["control", "treatment"]:
        g = df[df["group"] == group]
        row = {"group": group}
        prev = len(g)
        for stage in stages:
            count = g[stage].sum()
            row[f"{stage}_count"] = int(count)
            row[f"{stage}_rate_from_start"] = round(count / len(g), 4)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "ab_test_data.csv")

    print("=== SRM Check ===")
    print(check_sample_ratio_mismatch(df))

    print("\n=== Power Analysis (was this test adequately powered?) ===")
    print(power_analysis(baseline_rate=0.01, mde=0.003))

    print("\n=== Overall Conversion (Purchase) Test ===")
    print(two_proportion_ztest(df, "purchased"))

    print("\n=== Add-to-Cart Rate Test ===")
    print(two_proportion_ztest(df, "added_to_cart"))

    print("\n=== Revenue Test (Welch's t-test, purchasers only) ===")
    print(revenue_ttest(df))

    print("\n=== Day-7 Retention Test (purchasers only) ===")
    print(two_proportion_ztest(df[df["purchased"]], "retained_day7"))

    print("\n=== Segment Analysis: by Device ===")
    print(segment_analysis(df, "device", "purchased"))

    print("\n=== Funnel Drop-off ===")
    print(funnel_dropoff(df))
