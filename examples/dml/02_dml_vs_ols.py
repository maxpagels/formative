"""
Double ML vs OLS — non-linear confounding
=========================================
ability affects both education and income through ability², i.e. both
very low and very high ability raise them. OLS controls for ability as a
straight line, misses the curve, and stays biased. Double ML learns the
curve with random forests and recovers the true effect.

Requires the dml extra: pip install 'formative-ds[dml]'
"""

import numpy as np
import pandas as pd

from formative.causal import DAG, DoubleML, OLSObservational

RNG = np.random.default_rng(1)
N = 2_000
TRUE_EFFECT = 2.0

# ── 1. Simulate data with curved confounding ──────────────────────────────────
ability = RNG.normal(size=N)
education = 0.5 * ability**2 + RNG.normal(size=N)
income = TRUE_EFFECT * education + 2.0 * ability**2 + RNG.normal(size=N)

df = pd.DataFrame({"ability": ability, "education": education, "income": income})

dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

# ── 2. Estimate both ways ─────────────────────────────────────────────────────
ols = OLSObservational(dag, treatment="education", outcome="income").fit(df)
dml = DoubleML(dag, treatment="education", outcome="income").fit(df)

# ── 3. Compare ────────────────────────────────────────────────────────────────
print(f"True effect         : {TRUE_EFFECT:.4f}")
print(f"OLS estimate        : {ols.effect:.4f}  (bias: {ols.effect - TRUE_EFFECT:+.4f})")
print(f"Double ML estimate  : {dml.effect:.4f}  (bias: {dml.effect - TRUE_EFFECT:+.4f})")
