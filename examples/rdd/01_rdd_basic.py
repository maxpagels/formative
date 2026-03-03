"""
Basic RDD example: effect of crossing a test score threshold on an outcome.

DAG:
    score     → treatment
    score     → outcome
    treatment → outcome

Students scoring at or above 0 receive additional support (treatment = 1).
The running variable (score) also directly affects the outcome (e.g. higher
achievers tend to do better regardless of the support).

Under the RDD continuity assumption, students just above and just below the
threshold are locally comparable — the jump in outcome at the cutoff
identifies the Local Average Treatment Effect (LATE).

The true LATE is 2.0.
"""

import numpy as np
import pandas as pd

from formative import DAG, RDD

RNG = np.random.default_rng(42)
N = 2_000
TRUE_LATE = 2.0
CUTOFF = 0.0

score = RNG.uniform(-1, 1, size=N)

# DGP: score affects outcome directly (slope = 1.0); treatment adds 2.0
outcome = (
    1.0 * score
    + TRUE_LATE * (score >= CUTOFF).astype(float)
    + RNG.normal(scale=0.3, size=N)
)

# Treatment is NOT in the dataframe — RDD derives it from the threshold rule
df = pd.DataFrame({"score": score, "outcome": outcome})

dag = DAG()
dag.assume("score").causes("treatment", "outcome")
dag.assume("treatment").causes("outcome")

print(dag)
print()

# ── Full-sample RDD ────────────────────────────────────────────────────────────
result = RDD(dag, treatment="treatment", running_var="score",
             cutoff=CUTOFF, outcome="outcome").fit(df)

print(result.summary())
print(result.executive_summary())
print(result.statsmodels_result.summary())

# ── Bandwidth-restricted RDD ───────────────────────────────────────────────────
result_bw = RDD(dag, treatment="treatment", running_var="score",
                cutoff=CUTOFF, outcome="outcome", bandwidth=0.5).fit(df)

print("\n--- With bandwidth = 0.5 ---")
print(result_bw.summary())

# ── Refutation ─────────────────────────────────────────────────────────────────
report = result.refute(df)
print(report.summary())
