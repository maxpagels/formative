"""
Propensity Score Matching — bias correction
============================================
Shows how PSM corrects for confounding bias by comparing the ATT
against a naive (unadjusted) mean difference.
"""

import numpy as np
import pandas as pd
from formative import DAG, PropensityScoreMatching

RNG = np.random.default_rng(1)
N = 3_000
TRUE_ATT = 2.0

# ── 1. Simulate data with strong confounding ──────────────────────────────────
# ability raises both the chance of getting education AND income directly,
# so a naive comparison over-estimates the effect of education.
ability   = RNG.normal(size=N)
ps_latent = 0.8 * ability + RNG.normal(scale=0.5, size=N)   # strong PS link
education = (ps_latent > np.median(ps_latent)).astype(float)
income    = TRUE_ATT * education + 1.5 * ability + RNG.normal(size=N)

df = pd.DataFrame({"ability": ability, "education": education, "income": income})

# ── 2. DAG and estimation ─────────────────────────────────────────────────────
dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

result = PropensityScoreMatching(
    dag, treatment="education", outcome="income"
).fit(df)

# ── 3. Compare estimates ──────────────────────────────────────────────────────
print(result.summary())
print(f"True ATT            : {TRUE_ATT:.4f}")
print(f"ATT estimate        : {result.effect:.4f}  (bias: {result.effect - TRUE_ATT:+.4f})")
print(f"Unadjusted estimate : {result.unadjusted_effect:.4f}  (bias: {result.unadjusted_effect - TRUE_ATT:+.4f})")
