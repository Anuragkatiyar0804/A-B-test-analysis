"""
app.py
------
Two modes:
1. Demo — the fixed simulated experiment (original behavior).
2. Upload your own data — user provides a CSV, maps columns, gets the
   same statistical workflow run on their data.

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
from src.generic_stats import (
    check_srm,
    proportion_test,
    continuous_ttest,
    segment_breakdown,
)

DATA_PATH = Path(__file__).parent / "data" / "ab_test_data.csv"

st.set_page_config(page_title="A/B Test Analysis Tool", page_icon="📊", layout="wide")
st.title("📊 A/B Test Analysis Tool")
st.caption(
    "Run a proper A/B test analysis workflow — sample ratio mismatch check, "
    "significance testing, and segment breakdown — on the demo dataset or "
    "your own experiment data."
)

mode = st.radio("Choose a dataset", ["Try the demo experiment", "Upload my own CSV"], horizontal=True)
st.divider()

# ============================================================
# MODE 1: DEMO
# ============================================================
if mode == "Try the demo experiment":

    @st.cache_data
    def load_demo():
        return pd.read_csv(DATA_PATH)

    df = load_demo()

    srm = check_sample_ratio_mismatch(df)
    if srm["srm_detected"]:
        st.error(f"⚠️ Sample Ratio Mismatch detected (p={srm['p_value']}).")
    else:
        st.success(
            f"✅ No Sample Ratio Mismatch detected (p={srm['p_value']}, "
            f"observed split {srm['observed_ratio']:.1%} control / "
            f"{1 - srm['observed_ratio']:.1%} treatment)."
        )

    col1, col2, col3 = st.columns(3)
    conv_result = two_proportion_ztest(df, "purchased")
    with col1:
        st.metric("Purchase Conversion Lift", f"{conv_result['relative_lift_pct']}%", delta=f"p={conv_result['p_value']}")
        st.caption(f"Control: {conv_result['rate_control']:.4f} → Treatment: {conv_result['rate_treatment']:.4f}")

    rev_result = revenue_ttest(df)
    with col2:
        st.metric("Avg Revenue per Purchaser", f"₹{rev_result['diff']:+.2f}", delta=f"p={rev_result['p_value']}")
        st.caption(f"Control: ₹{rev_result['mean_revenue_control']} → Treatment: ₹{rev_result['mean_revenue_treatment']}")

    retention_result = two_proportion_ztest(df[df["purchased"]], "retained_day7")
    with col3:
        st.metric("Day-7 Retention (purchasers)", f"{retention_result['relative_lift_pct']}%", delta=f"p={retention_result['p_value']}")

    st.divider()
    st.subheader("Funnel: Viewed → Added to Cart → Purchased")
    st.dataframe(funnel_dropoff(df), use_container_width=True)
    st.image(str(Path(__file__).parent / "outputs" / "funnel_comparison.png"))

    st.divider()
    st.subheader("Segment Analysis (device)")
    st.dataframe(segment_analysis(df, "device", "purchased"), use_container_width=True)
    st.image(str(Path(__file__).parent / "outputs" / "segment_breakdown.png"))

    st.info(
        "The overall lift is driven primarily by the mobile segment (statistically "
        "significant), while desktop shows a smaller, non-significant effect."
    )

# ============================================================
# MODE 2: USER UPLOAD
# ============================================================
else:
    st.subheader("Upload your experiment data")
    st.caption(
        "CSV should have, at minimum: a column identifying which group each "
        "row belongs to (e.g. 'control'/'treatment'), and one binary outcome "
        "column (0/1 or True/False, e.g. 'converted')."
    )

    uploaded_file = st.file_uploader("Choose a CSV file (large files may take a moment on first load)", type="csv")

    if uploaded_file is not None:
        # Key on filename + size, NOT file bytes — hashing the full byte
        # content of a large file on every rerun (which happens on every
        # dropdown click) was the actual cause of the freezing. This way,
        # the CSV is only parsed once per genuinely new file, and every
        # subsequent rerun just reuses the already-parsed DataFrame from
        # session_state at near-zero cost.
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.get("_uploaded_file_id") != file_id:
            with st.spinner("Reading CSV — this only happens once per file..."):
                try:
                    parsed_df = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"Couldn't read that file: {e}")
                    st.stop()

                MAX_ROWS = 100_000
                if len(parsed_df) > MAX_ROWS:
                    st.warning(
                        f"This file has {len(parsed_df):,} rows. To keep things fast, "
                        f"analysis will run on a random sample of {MAX_ROWS:,} rows."
                    )
                    parsed_df = parsed_df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)

                st.session_state["_uploaded_file_id"] = file_id
                st.session_state["_uploaded_df"] = parsed_df

        user_df = st.session_state["_uploaded_df"]

        st.write("Preview:")
        st.dataframe(user_df.head(10), use_container_width=True)

        columns = list(user_df.columns)

        st.markdown("### Map your columns")
        c1, c2 = st.columns(2)
        with c1:
            group_col = st.selectbox("Which column identifies the group?", columns)
        with c2:
            metric_col = st.selectbox("Which column is your main binary outcome (0/1)?", columns)

        group_values = user_df[group_col].dropna().unique().tolist()
        if len(group_values) != 2:
            st.error(
                f"The group column '{group_col}' has {len(group_values)} unique values "
                f"({group_values}). This tool expects exactly 2 groups (e.g. control/treatment)."
            )
            st.stop()

        c3, c4 = st.columns(2)
        with c3:
            group_a = st.selectbox("Which value is the baseline / control group?", group_values, index=0)
        with c4:
            group_b_options = [v for v in group_values if v != group_a]
            group_b = st.selectbox("Which value is the treatment group?", group_b_options, index=0)

        expected_ratio = st.slider("Expected split ratio (control : total)", 0.1, 0.9, 0.5, 0.05)

        continuous_col = st.selectbox(
            "Optional: a continuous metric to compare (e.g. revenue) — or None",
            ["None"] + columns,
        )
        segment_col = st.selectbox(
            "Optional: a column to break results down by (e.g. device, country) — or None",
            ["None"] + columns,
        )

        if st.button("Run analysis"):
            st.divider()

            # --- SRM check ---
            srm = check_srm(user_df, group_col, expected_ratio)
            if "error" in srm:
                st.error(srm["error"])
                st.stop()

            if srm["srm_detected"]:
                st.error(
                    f"⚠️ Sample Ratio Mismatch detected (p={srm['p_value']}). "
                    "Treat any results below with caution."
                )
            else:
                st.success(f"✅ No Sample Ratio Mismatch detected (p={srm['p_value']}).")

            # --- Primary metric ---
            try:
                result = proportion_test(user_df, group_col, metric_col, group_a, group_b)
                st.subheader(f"Primary metric: {metric_col}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Relative lift", f"{result['relative_lift_pct']}%")
                m2.metric("p-value", f"{result['p_value']}")
                m3.metric("Significant at 0.05?", "Yes" if result["significant_at_05"] else "No")
                st.caption(
                    f"{group_a}: {result[f'rate_{group_a}']} → {group_b}: {result[f'rate_{group_b}']} "
                    f"| 95% CI on difference: {result['ci_95_diff']}"
                )
            except Exception as e:
                st.error(f"Couldn't run the proportion test — check that '{metric_col}' is binary (0/1). Error: {e}")

            # --- Continuous metric ---
            if continuous_col != "None":
                try:
                    cont_result = continuous_ttest(user_df, group_col, continuous_col, group_a, group_b)
                    st.subheader(f"Continuous metric: {continuous_col}")
                    st.write(cont_result)
                except Exception as e:
                    st.error(f"Couldn't run the t-test on '{continuous_col}': {e}")

            # --- Segment breakdown ---
            if segment_col != "None":
                try:
                    seg_df = segment_breakdown(user_df, group_col, metric_col, segment_col, group_a, group_b)
                    st.subheader(f"Segment breakdown by {segment_col}")
                    if seg_df.empty:
                        st.warning("No valid segments found (each segment needs both groups present).")
                    else:
                        st.dataframe(seg_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Couldn't run segment breakdown: {e}")
    else:
        st.info("Upload a CSV above to get started, or switch to the demo tab to see an example first.")

st.divider()
st.caption(
    "Built to demonstrate a full experimentation workflow: SRM checking, "
    "significance testing, and segment-level analysis — not just a single "
    "p-value."
)