"""
IV vs OLS: showing confounding bias and how IV corrects it.

When ability is unobserved, OLS conflates the effect of education with
the effect of ability, producing a biased estimate. IV recovers the
true effect by using proximity as an instrument.

DAG:
    proximity → education → income
                  ↑              ↑
    ability ──────┘──────────────┘

True effect of education on income: 2.0
"""

import numpy as np
import pandas as pd

from formative import DAG, IV2SLS

RNG = np.random.default_rng(0)
N = 5_000

proximity = RNG.normal(size=N)
ability   = RNG.normal(size=N)
education = 0.5 * proximity + 0.5 * ability + RNG.normal(size=N)
income    = 2.0 * education + 0.8 * ability + RNG.normal(size=N)

df = pd.DataFrame({"proximity": proximity, "education": education, "income": income})

dag = DAG()
dag.assume("proximity").causes("education")
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

result = IV2SLS(
    dag, treatment="education", outcome="income", instrument="proximity"
).fit(df)

print(result.summary())
