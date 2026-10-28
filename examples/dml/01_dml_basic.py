"""
Double ML — basic example
=========================
Estimate the effect of a continuous treatment (education) on income
whilst adjusting for a confounder (ability) with machine-learning models.

Requires the dml extra: pip install 'formative-ds[dml]'
"""

import numpy as np
import pandas as pd

from formative.causal import DAG, DoubleML

RNG = np.random.default_rng(0)
N = 2_000

# ── 1. Simulate data ──────────────────────────────────────────────────────────
ability = RNG.normal(size=N)
education = 0.5 * ability**2 + RNG.normal(size=N)
income = 2.0 * education + 2.0 * ability**2 + RNG.normal(size=N)

df = pd.DataFrame({"ability": ability, "education": education, "income": income})

# ── 2. Declare causal assumptions ─────────────────────────────────────────────
dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

# ── 3. Estimate via Double ML ─────────────────────────────────────────────────
result = DoubleML(dag, treatment="education", outcome="income").fit(df)

print(result.summary())
print(result.executive_summary())
