"""
Basic OLS example: effect of education on income.

DAG:
    ability → education → income
    ability ↗            ↗
           ──────────────

ability is a confounder: it causes both education and income.
Because it is in the dataframe, OLSObservational controls for it automatically.
The true causal effect of education on income is 2.0.
"""

import numpy as np
import pandas as pd

from formative import DAG, OLSObservational

RNG = np.random.default_rng(0)
N = 2_000

ability    = RNG.normal(size=N)
education  = 0.5 * ability + RNG.normal(size=N)
income     = 2.0 * education + 0.8 * ability + RNG.normal(size=N)

df = pd.DataFrame({"ability": ability, "education": education, "income": income})

dag = DAG()
dag.causes("ability", "education")
dag.causes("ability", "income")
dag.causes("education", "income")

print(dag)
print()

result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
print(result.summary())
