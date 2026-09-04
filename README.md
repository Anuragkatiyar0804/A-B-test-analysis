# A/B Test Analysis: Product Feature Rollout

## Business Problem

Product and growth teams run experiments constantly — a new onboarding
flow, a recommendation widget, a pricing change — but the actual decision
that matters ("should we ship this to everyone?") often gets made off a
single headline number: "conversion went up 40%, ship it." That headline
number can be misleading in a few specific ways:

- The experiment's group assignment could be broken (a logging bug, a
  redirect issue) without anyone noticing, making the whole comparison
  invalid.
- The test might not have collected enough data to actually detect the
  effect size that matters, making a "no significant difference" result
  look like "no effect" when it's really "inconclusive."
- The effect might be concentrated in one user segment and absent or
  reversed in another, so "ship to everyone" is a worse decision than
  "ship to the segment where it actually works."

I built this project to work through that full decision process myself —
not just compute a p-value, but build the checks a real experiment needs
before that p-value can be trusted, and the segment-level analysis that
turns a stat into an actual product recommendation.

## How I Approached It

Since real company experiment data isn't public, I generated a synthetic
dataset for a simulated feature rollout (a new onboarding/recommendation
widget), with known effects built into the data generator itself. That
let me confirm my analysis pipeline correctly recovers the true signal
before trusting it on anything else — a way of testing the analysis code,
not just the data.

I then ran the same pipeline on a real, public marketing A/B test dataset
from Kaggle to check that it holds up outside a dataset I built to make
myself look good. It correctly found a real, significant lift in that
data and correctly flagged a genuine sample ratio mismatch that exists in
the real dataset's group sizes — both useful signals that the tool
generalizes, not just fits one demo.

Finally, I extended it from a fixed one-off analysis into a general tool:
anyone can upload their own experiment CSV, map their own column names,
and run the same statistical workflow on their own data.

## What I Did, Step by Step

**1. Checked the experiment was even valid before trusting anything.**
A Sample Ratio Mismatch (SRM) check, using a chi-square test, compares
the actual observed group split against the intended split. If they don't
match, something in the assignment or logging pipeline is likely broken,
and no result downstream can be trusted until that's investigated.

**2. Checked whether the experiment had enough data.**
A power analysis calculates the minimum sample size needed to reliably
detect a meaningful effect size. This matters because a "not significant"
result only means "no effect" if the test was adequately powered in the
first place — otherwise it just means "inconclusive."

**3. Tested the primary metric properly.**
A one-sided two-proportion z-test on purchase conversion, since the
hypothesis is directional (the feature should increase conversion, not
just change it in either direction).

**4. Tested secondary metrics with the right test for each.**
Add-to-cart rate (another proportion test), revenue per purchaser
(Welch's t-test, which doesn't assume equal variance between groups), and
day-7 retention among purchasers.

**5. Broke the primary result down by segment.**
This is the step I think matters most. The pooled result showed a 43%
lift — but splitting by device showed the effect was statistically
significant for mobile users and not significant for desktop. A "ship to
everyone" recommendation based on the pooled number alone would have been
less accurate than a segment-aware one: ship to mobile now, and either
extend the desktop test or investigate why the effect doesn't transfer
before deciding there too.

**6. Validated on real data.**
Ran the full pipeline on a public Kaggle marketing dataset (ad vs. PSA
groups) to confirm the tool works on real-world data, not just data I
generated myself.

**7. Generalized it into a tool.**
Built an "upload your own CSV" mode with column mapping, so the same
workflow can run on any two-group experiment dataset, not just this one.

## Results (on the simulated experiment)

- **Primary metric (purchase conversion):** control 0.97% → treatment
  1.38%, a 43.0% relative lift, p = 0.00006
- **Revenue per purchaser:** +₹2.42 average order value, p = 0.035
- **Day-7 retention:** no significant effect, p = 0.60
- **Segment breakdown:** mobile +54.5% lift (p = 0.000084, significant),
  desktop +24.1% lift (p = 0.0847, not significant)

## Project Structure

```
data/generate_data.py    → generates the synthetic dataset with known
                            ground-truth effects baked in
src/stats_tests.py       → SRM check, power analysis, two-proportion
                            z-test, Welch's t-test, segment analysis,
                            funnel analysis (built for the fixed demo
                            dataset's schema)
src/generic_stats.py     → column-agnostic versions of the same tests,
                            for the upload-your-own-data mode
src/visualize.py         → generates the funnel/conversion/segment charts
app.py                   → Streamlit app — demo mode and upload mode
outputs/                 → generated charts (PNG)
```

## How to Run It

```bash
pip install -r requirements.txt
python data/generate_data.py     # generates data/ab_test_data.csv
python src/stats_tests.py        # prints the full analysis to console
python src/visualize.py          # generates outputs/*.png
streamlit run app.py             # interactive dashboard, both modes
```

## Design Choices Worth Noting

- **SRM check runs first, before anything else** — an experiment with
  broken randomization invalidates every result downstream, so this
  isn't optional pre-work.
- **Welch's t-test, not Student's**, for the revenue comparison — doesn't
  assume equal variance between groups, and costs almost nothing in
  power if variances happen to be equal.
- **One-sided z-test for the primary metric** — the hypothesis is
  directional, which is slightly more powerful than an unjustified
  two-sided test.
- **Segment analysis isn't a bonus feature** — in this case, it changed
  the actual recommendation from "ship to everyone" to "ship to mobile."
- **Upload mode is cached via `st.session_state`, not by hashing file
  bytes** — an earlier version re-hashed the full uploaded file on every
  single dropdown interaction (since Streamlit reruns the whole script
  on every widget change), which froze the app on large files. Caching
  by filename + size instead fixed it, since the expensive parse only
  needs to happen once per genuinely new file.

## What I'd Add With More Time

- Sequential testing correction (this analysis assumes a fixed sample
  size decided in advance, with no early peeking at results)
- Multiple comparison correction for the segment analysis, since testing
  several segments raises the chance of a false positive in at least one
- CUPED variance reduction using pre-experiment covariates, to detect
  smaller effects at the same sample size
