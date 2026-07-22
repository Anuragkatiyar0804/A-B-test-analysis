"""
generic_stats.py
----------------
Column-agnostic versions of the core tests, for use when a user uploads
their own CSV with arbitrary column names and group labels (unlike
src/stats_tests.py, which is written specifically for the demo dataset's
fixed schema).
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def check_srm(df: pd.DataFrame, group_col: str, expected_ratio: float = 0.5) -> dict:
    counts = df[group_col].value_counts()
    if len(counts) != 2:
        return {"error": f"Expected exactly 2 groups in '{group_col}', found {len(counts)}."}

    labels = list(counts.index)
    n1, n2 = counts[labels[0]], counts[labels[1]]
    total = n1 + n2

    chi2, p_value = stats.chisquare([n1, n2], f_exp=[total * expected_ratio, total * (1 - expected_ratio)])

    return {
        "labels": labels,
        "counts": {labels[0]: int(n1), labels[1]: int(n2)},
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "srm_detected": p_value < 0.01,
    }


def proportion_test(df: pd.DataFrame, group_col: str, metric_col: str,
                     group_a: str, group_b: str) -> dict:
    """
    Two-proportion z-test. group_b is treated as the 'treatment' (the one
    expected to have a higher rate) for the one-sided test direction.
    metric_col must be binary (0/1, True/False).
    """
    a = df[df[group_col] == group_a]
    b = df[df[group_col] == group_b]

    successes = np.array([b[metric_col].sum(), a[metric_col].sum()])
    nobs = np.array([len(b), len(a)])

    z_stat, p_value = proportions_ztest(successes, nobs, alternative="two-sided")

    rate_a = a[metric_col].mean()
    rate_b = b[metric_col].mean()
    diff = rate_b - rate_a
    relative_lift = (diff / rate_a * 100) if rate_a != 0 else float("nan")

    se_diff = np.sqrt(rate_a * (1 - rate_a) / len(a) + rate_b * (1 - rate_b) / len(b))
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "metric": metric_col,
        f"rate_{group_a}": round(rate_a, 5),
        f"rate_{group_b}": round(rate_b, 5),
        "absolute_diff": round(diff, 5),
        "relative_lift_pct": round(relative_lift, 2),
        "ci_95_diff": (round(ci_low, 5), round(ci_high, 5)),
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": p_value < 0.05,
    }


def continuous_ttest(df: pd.DataFrame, group_col: str, value_col: str,
                      group_a: str, group_b: str) -> dict:
    """Welch's t-test for a continuous metric (e.g. revenue, session time)."""
    a = df[df[group_col] == group_a][value_col].dropna()
    b = df[df[group_col] == group_b][value_col].dropna()

    t_stat, p_value = stats.ttest_ind(b, a, equal_var=False)

    return {
        "metric": value_col,
        f"mean_{group_a}": round(a.mean(), 4),
        f"mean_{group_b}": round(b.mean(), 4),
        "diff": round(b.mean() - a.mean(), 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": p_value < 0.05,
        f"n_{group_a}": len(a),
        f"n_{group_b}": len(b),
    }


def segment_breakdown(df: pd.DataFrame, group_col: str, metric_col: str,
                       segment_col: str, group_a: str, group_b: str) -> pd.DataFrame:
    rows = []
    for seg_value in df[segment_col].dropna().unique():
        seg_df = df[df[segment_col] == seg_value]
        if seg_df[group_col].nunique() < 2:
            continue
        r = proportion_test(seg_df, group_col, metric_col, group_a, group_b)
        r[segment_col] = seg_value
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)