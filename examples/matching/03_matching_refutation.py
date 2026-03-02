"""
Propensity Score Matching — refutation checks
==============================================
After fitting, run refutation checks to probe whether the ATT estimate
is stable and not spurious.

  - Placebo treatment: permuting treatment labels should yield ~0 effect.
  - Random common cause: adding a noise covariate should not shift the ATT.
"""

import numpy as np
import pandas as pd
from formative import DAG, PropensityScoreMatching

RNG = np.random.default_rng(2)
N = 3_000

# ── 1. Well-specified data (checks should pass) ───────────────────────────────
ability   = RNG.normal(size=N)
ps_latent = 0.5 * ability + RNG.normal(scale=0.5, size=N)
education = (ps_latent > np.median(ps_latent)).astype(float)
income    = 2.0 * education + 0.8 * ability + RNG.normal(size=N)

df = pd.DataFrame({"ability": ability, "education": education, "income": income})

dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

result = PropensityScoreMatching(
    dag, treatment="education", outcome="income"
).fit(df)

print("=== Original estimate ===")
print(result.summary())
print(result.executive_summary())

print("=== Refutation report ===")
report = result.refute(df)
print(report.summary())
