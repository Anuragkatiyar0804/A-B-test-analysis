"""
generate_data.py
----------------
Generates a synthetic but realistic user-level A/B test dataset simulating
a product feature rollout (e.g., a new onboarding flow / recommendation
widget).

Why synthetic data, and why this isn't "cheating":
Real company A/B test data is never public (it's commercially sensitive),
so every portfolio project in this space uses simulated data. What makes
this defensible in an interview is that we KNOW the ground truth effect
we baked in, so we can verify our statistical pipeline actually recovers
the true effect — proving the analysis code is correct rather than just
producing plausible-looking numbers.

Deliberate realism baked in (be ready to explain each in interviews):
1. Unequal group sizes are avoided at assignment time, but a small random
   dropout is added AFTER assignment — this creates a mild, realistic risk
   of Sample Ratio Mismatch (SRM) that the analysis must check for.
2. A segment-level heterogeneous treatment effect: the new feature helps
   mobile users more than desktop users. Naive pooled analysis can mask or
   distort this (Simpson's-paradox-adjacent) — good analysis must segment.
3. A weekday/weekend seasonality effect on conversion, independent of
   treatment — tests whether the analysis correctly attributes lift to
   treatment vs. confounding by day-of-week (mitigated by randomization,
   but worth showing you checked).
4. A modest revenue effect correlated with, but distinct from, conversion
   (treatment increases conversion rate AND average order value slightly)
   so t-test and proportion-test tell different parts of the story.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_USERS = 40000
OUT_PATH = Path(__file__).parent / "ab_test_data.csv"

# Ground truth parameters (unknown to the "analyst" persona, known to us
# for validation purposes)
BASE_CONVERSION = {"desktop": 0.070, "mobile": 0.055}
TREATMENT_LIFT = {"desktop": 0.003, "mobile": 0.018}  # mobile benefits more
BASE_AOV = 45.0          # average order value, INR-equivalent units scaled
TREATMENT_AOV_LIFT = 2.5
RETENTION_BASE = 0.28    # day-7 retention among converters
RETENTION_LIFT = 0.04


def generate():
    rng = np.random.default_rng(RNG_SEED)

    user_id = np.arange(1, N_USERS + 1)
    group = rng.choice(["control", "treatment"], size=N_USERS, p=[0.5, 0.5])
    device = rng.choice(["mobile", "desktop"], size=N_USERS, p=[0.65, 0.35])
    signup_day = rng.integers(0, 14, size=N_USERS)  # 14-day experiment window
    day_of_week = signup_day % 7
    is_weekend = np.isin(day_of_week, [5, 6])

    # --- Simulate a small SRM-inducing dropout ---
    # e.g. a logging bug drops slightly more treatment-mobile events
    drop_mask = (group == "treatment") & (device == "mobile") & (rng.random(N_USERS) < 0.015)
    keep_mask = ~drop_mask

    # --- Conversion probability ---
    base_conv = np.where(device == "mobile", BASE_CONVERSION["mobile"], BASE_CONVERSION["desktop"])
    lift = np.where(group == "treatment",
                     np.where(device == "mobile", TREATMENT_LIFT["mobile"], TREATMENT_LIFT["desktop"]),
                     0.0)
    weekend_effect = np.where(is_weekend, 0.006, 0.0)  # slightly higher conversion on weekends
    conv_prob = np.clip(base_conv + lift + weekend_effect, 0, 1)
    converted = rng.random(N_USERS) < conv_prob

    # --- Funnel stages (viewed -> added_to_cart -> purchased) ---
    viewed = np.ones(N_USERS, dtype=bool)  # everyone in the experiment viewed the product page
    add_to_cart_prob = np.clip(conv_prob * rng.uniform(2.2, 2.8, N_USERS), 0, 0.95)
    added_to_cart = rng.random(N_USERS) < add_to_cart_prob
    purchased = added_to_cart & converted  # purchase requires both funnel progression and conversion draw

    # --- Revenue (only for purchasers) ---
    aov_lift = np.where(group == "treatment", TREATMENT_AOV_LIFT, 0.0)
    revenue = np.where(
        purchased,
        rng.normal(BASE_AOV + aov_lift, 12).clip(min=5),
        0.0
    )

    # --- Day-7 retention (only defined for purchasers) ---
    retention_prob = np.where(
        group == "treatment",
        RETENTION_BASE + RETENTION_LIFT,
        RETENTION_BASE
    )
    retained_day7 = np.where(purchased, rng.random(N_USERS) < retention_prob, False)

    df = pd.DataFrame({
        "user_id": user_id,
        "group": group,
        "device": device,
        "signup_day": signup_day,
        "is_weekend": is_weekend,
        "viewed_product": viewed,
        "added_to_cart": added_to_cart,
        "purchased": purchased,
        "revenue": revenue.round(2),
        "retained_day7": retained_day7,
    })

    df = df[keep_mask].reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Generated {len(df)} rows (dropped {drop_mask.sum()} for simulated SRM)")
    print(f"Saved to {OUT_PATH}")
    print("\nGroup sizes:")
    print(df["group"].value_counts())
    print("\nOverall conversion rate by group:")
    print(df.groupby("group")["purchased"].mean())
    return df


if __name__ == "__main__":
    generate()
