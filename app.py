"""
app.py
------
Interactive dashboard for exploring the A/B test results.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.stats_tests import (
    check_sample_ratio_mismatch,
    two_proportion_ztest,
    revenue_ttest,
    segment_analysis,
    funnel_dropoff,
)

DATA_PATH = Path(__file__).parent / "data" / "ab_test_data.csv"

st.set_page_config(page_title="A/B Test Analysis Dashboard", page_icon="📊", layout="wide")
st.title("📊 A/B Test Analysis: Product Feature Rollout")
st.caption(
    "Simulated experiment: new onboarding/recommendation feature — measuring "
    "impact on conversion, revenue, and retention, with proper experimental rigor."
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

# --- SRM check banner ---
srm = check_sample_ratio_mismatch(df)
if srm["srm_detected"]:
    st.error(
        f"⚠️ Sample Ratio Mismatch detected (p={srm['p_value']}). "
        "Results below should NOT be trusted until this is investigated."
    )
else:
    st.success(
        f"✅ No Sample Ratio Mismatch detected (p={srm['p_value']}, "
        f"observed split {srm['observed_ratio']:.1%} control / "
        f"{1 - srm['observed_ratio']:.1%} treatment). Safe to proceed with analysis."
    )

st.divider()

# --- Top-line metrics ---
col1, col2, col3 = st.columns(3)

conv_result = two_proportion_ztest(df, "purchased")
with col1:
    st.metric(
        "Purchase Conversion Lift",
        f"{conv_result['relative_lift_pct']}%",
        delta=f"p={conv_result['p_value']}",
    )
    st.caption(f"Control: {conv_result['rate_control']:.4f} → Treatment: {conv_result['rate_treatment']:.4f}")

rev_result = revenue_ttest(df)
with col2:
    st.metric(
        "Avg Revenue per Purchaser",
        f"₹{rev_result['diff']:+.2f}",
        delta=f"p={rev_result['p_value']}",
    )
    st.caption(f"Control: ₹{rev_result['mean_revenue_control']} → Treatment: ₹{rev_result['mean_revenue_treatment']}")

retention_result = two_proportion_ztest(df[df["purchased"]], "retained_day7")
with col3:
    st.metric(
        "Day-7 Retention (purchasers)",
        f"{retention_result['relative_lift_pct']}%",
        delta=f"p={retention_result['p_value']}",
    )
    st.caption("Not statistically significant — see note below" if not retention_result["significant_at_05"] else "Significant")

st.divider()

# --- Funnel ---
st.subheader("Funnel: Viewed → Added to Cart → Purchased")
funnel = funnel_dropoff(df)
st.dataframe(funnel, use_container_width=True)
st.image(str(Path(__file__).parent / "outputs" / "funnel_comparison.png"))

st.divider()

# --- Segment analysis ---
st.subheader("Segment Analysis: Is the effect driven by one segment?")
segment_col = st.selectbox("Segment by", ["device"])
seg_df = segment_analysis(df, segment_col, "purchased")
st.dataframe(seg_df, use_container_width=True)
st.image(str(Path(__file__).parent / "outputs" / "segment_breakdown.png"))

st.info(
    "**Business takeaway:** the overall lift is being driven primarily by the "
    "mobile segment (statistically significant), while desktop shows a smaller, "
    "non-significant effect. A recommendation to 'ship to everyone' vs. "
    "'ship to mobile only' should be made with this breakdown in mind, not "
    "just the pooled top-line number."
)

st.divider()
st.caption(
    "Built as a portfolio project demonstrating experimental design rigor: "
    "SRM checking, power analysis, proper test selection (z-test/Welch's t-test), "
    "and segment-level analysis to catch effects a pooled result would hide."
)
