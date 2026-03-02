"""
IV refutation: testing instrument strength with the first-stage F-statistic.

A weak instrument (low correlation with treatment) causes IV estimates to be
severely biased toward OLS and to have unreliable confidence intervals.
The conventional threshold is F ≥ 10 (Stock & Yogo, 2005).

This example runs the same DAG and estimation twice:
  1. Strong instrument (proximity coefficient 0.5) → refutation passes.
  2. Weak instrument  (proximity coefficient 0.02) → refutation fails.
"""

import numpy as np
import pandas as pd

from formative import DAG, IV2SLS

RNG = np.random.default_rng(0)
N = 5_000


def make_data(instrument_strength: float):
    proximity = RNG.normal(size=N)
    ability   = RNG.normal(size=N)
    education = instrument_strength * proximity + 0.5 * ability + RNG.normal(size=N)
    income    = 2.0 * education + 0.8 * ability + RNG.normal(size=N)
    return pd.DataFrame({"proximity": proximity, "education": education, "income": income})


dag = DAG()
dag.assume("proximity").causes("education")
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

# ── Strong instrument ──────────────────────────────────────────────────────
print("=" * 52)
print("STRONG INSTRUMENT  (coefficient = 0.5)")
print("=" * 52)

df_strong = make_data(instrument_strength=0.5)
result_strong = IV2SLS(
    dag, treatment="education", outcome="income", instrument="proximity"
).fit(df_strong)
print(result_strong.summary())
print(result_strong.executive_summary())
print(result_strong.refute(df_strong).summary())

# ── Weak instrument ────────────────────────────────────────────────────────
print("=" * 52)
print("WEAK INSTRUMENT  (coefficient = 0.02)")
print("=" * 52)

df_weak = make_data(instrument_strength=0.02)
result_weak = IV2SLS(
    dag, treatment="education", outcome="income", instrument="proximity"
).fit(df_weak)
print(result_weak.summary())
print(result_weak.refute(df_weak).summary())
