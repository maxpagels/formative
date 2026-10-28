"""
Double ML — refutation checks
=============================
Binary treatment, so the interactive model (IRM) is used and all four
checks run:

  - Random common cause: adding a noise covariate should not shift the estimate.
  - Placebo treatment: a permuted treatment should show no effect.
  - Nuisance model fit: the outcome model should beat predicting the mean, and
    confounders should not almost fully determine treatment.
  - Overlap: few units should be (almost) certain to be treated or untreated.

The second dataset has much stronger selection on ability, so units with
extreme ability are nearly always treated and the overlap check fails.

Requires the dml extra: pip install 'formative-ds[dml]'
"""

import numpy as np
import pandas as pd

from formative.causal import DAG, DoubleML

N = 2_000

dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")


def simulate(selection: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ability = rng.normal(size=N)
    latent = selection * ability**2 + rng.normal(size=N)
    education = (latent > np.median(latent)).astype(float)
    income = 2.0 * education + 2.0 * ability**2 + rng.normal(size=N)
    return pd.DataFrame({"ability": ability, "education": education, "income": income})


# ── 1. Good overlap (checks should pass) ──────────────────────────────────────
df = simulate(selection=0.5, seed=2)
result = DoubleML(dag, treatment="education", outcome="income").fit(df)
print(result.summary())
print(result.refute(df).summary())

# ── 2. Poor overlap (overlap check should fail) ───────────────────────────────
df = simulate(selection=3.0, seed=2)
result = DoubleML(dag, treatment="education", outcome="income").fit(df)
print(result.refute(df).summary())
