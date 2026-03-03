"""
RDD example: effect of a safety regulation taking effect at a specific week.

DAG:
    week        → treatment
    week        → injuries
    treatment   → injuries

A factory records weekly injury counts. At week 100, a new safety regulation
takes effect (treatment = 1). The running variable (week) runs from 0 upwards.

Time itself has a mild downward trend in injuries (gradual process improvements),
but the regulation causes an additional sharp drop at the cutoff.

Under the RDD continuity assumption, weeks just before and just after week 100
are locally comparable — the jump in injuries at the cutoff identifies the
Local Average Treatment Effect (LATE).

The true LATE is -5.0 injuries per week.
"""

import numpy as np
import pandas as pd

from formative import DAG, RDD

RNG = np.random.default_rng(7)
N = 2_000
TRUE_LATE = -5.0
CUTOFF = 100.0

# Running variable: time steps (weeks) drawn uniformly from 0 to 200
week = RNG.uniform(0, 200, size=N)

# DGP: baseline injuries decline slowly over time (−0.02 per week);
# crossing the regulation cutoff drops injuries by 5
injuries = (
    20.0
    - 0.02 * week
    + TRUE_LATE * (week >= CUTOFF).astype(float)
    + RNG.normal(scale=1.0, size=N)
)

# Treatment is NOT in the dataframe — RDD derives it from the threshold rule
df = pd.DataFrame({"week": week, "injuries": injuries})

dag = DAG()
dag.assume("week").causes("treatment", "injuries")
dag.assume("treatment").causes("injuries")

print(dag)
print()

# ── Full-sample RDD ────────────────────────────────────────────────────────────
result = RDD(
    dag,
    treatment="treatment",
    running_var="week",
    cutoff=CUTOFF,
    outcome="injuries",
).fit(df)

print(result.summary())
print(result.executive_summary())
print(result.statsmodels_result.summary())

# ── Bandwidth-restricted RDD ───────────────────────────────────────────────────
result_bw = RDD(
    dag,
    treatment="treatment",
    running_var="week",
    cutoff=CUTOFF,
    outcome="injuries",
    bandwidth=20.0,
).fit(df)

print("\n--- With bandwidth = 20 (weeks 80–120) ---")
print(result_bw.summary())

# ── Refutation ─────────────────────────────────────────────────────────────────
report = result.refute(df)
print(report.summary())
