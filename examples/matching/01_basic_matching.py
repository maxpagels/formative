"""
Propensity Score Matching — basic example
==========================================
Estimate the ATT of a binary treatment (further education) on income
whilst adjusting for a confounding variable (ability) via matching.
"""

import numpy as np
import pandas as pd
from formative import DAG, PropensityScoreMatching

RNG = np.random.default_rng(0)
N = 3_000

# ── 1. Simulate data ──────────────────────────────────────────────────────────
ability   = RNG.normal(size=N)
ps_latent = 0.5 * ability + RNG.normal(scale=0.5, size=N)
education = (ps_latent > np.median(ps_latent)).astype(float)   # binary 0/1
income    = 2.0 * education + 0.8 * ability + RNG.normal(size=N)

df = pd.DataFrame({"ability": ability, "education": education, "income": income})

# ── 2. Declare causal assumptions ─────────────────────────────────────────────
dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

# ── 3. Estimate via PSM ───────────────────────────────────────────────────────
result = PropensityScoreMatching(
    dag, treatment="education", outcome="income"
).fit(df)

print(result.summary())
