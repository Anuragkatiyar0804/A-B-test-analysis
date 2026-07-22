# A/B Test Analysis: Product Feature Rollout

An end-to-end A/B test analysis project simulating a real product experiment
(new onboarding/recommendation feature), built to demonstrate proper
experimental design and statistical rigor — not just "ran a t-test and got
a p-value."

## Why this project

Built specifically to demonstrate skills required for Product Analyst /
Data Analyst roles involving experimentation: designing tests correctly,
validating experiment integrity before trusting results, and translating
statistical output into a business decision — including catching a case
where the pooled result would mislead you.

## Project structure

```
data/generate_data.py    → synthetic experiment data with known ground-truth
                           effects (so we can verify the analysis pipeline
                           actually recovers the true signal)
src/stats_tests.py       → SRM check, power analysis, two-proportion z-test,
                           Welch's t-test, segment analysis, funnel analysis
src/visualize.py         → generates charts (funnel, conversion, segments)
app.py                   → interactive Streamlit dashboard
outputs/                 → generated charts (PNG)
```

## How to run

```bash
pip install -r requirements.txt
python data/generate_data.py     # generates data/ab_test_data.csv
python src/stats_tests.py        # prints full analysis to console
python src/visualize.py          # generates outputs/*.png
streamlit run app.py             # interactive dashboard
```

## Methodology (in the order a real analysis should follow)

### 1. Sample Ratio Mismatch (SRM) check — run FIRST, always
Before trusting any effect size, verify the actual group split matches the
intended randomization (a chi-square goodness-of-fit test). If this fails,
**stop** — any downstream result could be an artifact of a broken
assignment/logging pipeline, not a real treatment effect.

**Result:** No SRM detected (p=0.62, observed split 49.9% / 50.1%). Safe to proceed.

### 2. Power analysis
Before (in a real scenario) launching the test, calculate the minimum
sample size needed to detect a meaningful effect at 80% power. This is
what determines how long an experiment needs to run — a step often skipped
by candidates who only know how to analyze data after the fact.

**Result:** To detect a 0.3pp lift on a ~1% baseline conversion rate at 80%
power, ~19,700 users per group are required. Our experiment had ~19,800-19,950
per group — right at the edge of adequate power, which is itself worth
flagging rather than treating the result as unambiguous.

### 3. Primary metric test: two-proportion z-test on purchase conversion
- Control: 0.97% → Treatment: 1.38%
- **Relative lift: +43.0%, p = 0.00006** (highly significant)
- 95% CI on absolute difference: [0.0021, 0.0063]

### 4. Secondary metrics
- **Add-to-cart rate:** +19.8% relative lift, p ≈ 0 (significant)
- **Revenue per purchaser (Welch's t-test):** +₹2.42 avg order value, p = 0.035 (significant)
- **Day-7 retention among purchasers:** -3.9% relative (not significant, p = 0.60) — the
  feature affects acquisition/conversion but shows no measurable retention
  effect yet in this window.

### 5. Segment analysis — the key finding
Breaking the primary result down by device type:

| Segment | Control rate | Treatment rate | Relative lift | p-value | Significant? |
|---|---|---|---|---|---|
| Mobile | 0.93% | 1.44% | +54.5% | 0.000084 | **Yes** |
| Desktop | 1.03% | 1.28% | +24.1% | 0.0847 | No |

**This is the finding that matters most for the business decision.** The
overall "43% lift" headline number is being driven almost entirely by
mobile users. Desktop shows a small, statistically inconclusive effect.
A pooled recommendation ("ship to everyone") would be less precise than
the segment-aware one ("ship to mobile now; extend desktop test duration
or investigate why the effect doesn't transfer before deciding there").

## Recommendation (how you'd phrase this to a PM in an interview)

> "The feature shows a statistically significant lift in purchase
> conversion and revenue per purchaser, but the effect is concentrated in
> mobile users — desktop's effect isn't distinguishable from noise at
> current sample size. I'd recommend shipping to mobile now, and either
> extending the desktop test to reach adequate power, or investigating
> why the mechanism doesn't transfer to desktop before deciding to roll
> out there too. Retention at day 7 shows no effect yet, which is worth
> watching in a longer follow-up window since a pure acquisition effect
> that doesn't stick isn't the full win it looks like at 2 weeks."

## Design decisions to discuss in interviews

1. **Why SRM check comes before everything else** — an experiment with a
   broken assignment pipeline invalidates every downstream number, so
   integrity checks are not optional pre-work.
2. **Why Welch's t-test, not Student's** — doesn't assume equal variance
   between groups; safer default, negligible power cost.
3. **Why one-sided z-test for the primary metric** — the hypothesis is
   directional (feature should *increase* conversion), which is slightly
   more powerful than an unjustified two-sided test.
4. **Why segment analysis isn't optional** — a pooled effect can hide
   (or be entirely driven by) one segment; the business decision changes
   materially depending on which is true.
5. **Why power analysis matters even after the fact** — it tells you
   whether a "not significant" result means "no effect" or "underpowered
   test," which are very different conclusions with different next steps.

## Possible extensions (mention as "next steps" if asked)
- Sequential testing / peeking correction (this analysis assumes a
  fixed sample size decided in advance — didn't peek early).
- Multiple comparison correction (Bonferroni/Benjamini-Hochberg) since we
  tested 4 metrics — worth flagging even though the primary metric was
  pre-registered.
- CUPED variance reduction using pre-experiment covariates for more power
  at the same sample size.
