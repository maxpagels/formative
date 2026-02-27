"""
Basic IV example: effect of education on income with an unobserved confounder.

DAG:
    proximity → education → income
                  ↑              ↑
    ability ──────┘──────────────┘

ability is an unobserved confounder (not in the dataframe). OLSObservational
would refuse to run. IV2SLS uses proximity to college as an instrument:
  - Relevance: proximity affects education (first stage)
  - Exclusion: proximity affects income only through education

The true causal effect of education on income is 2.0.
"""

import numpy as np
import pandas as pd

from formative import DAG, IV2SLS

RNG = np.random.default_rng(0)
N = 5_000

proximity = RNG.normal(size=N)          # instrument: exogenous
ability   = RNG.normal(size=N)          # unobserved confounder
education = 0.5 * proximity + 0.5 * ability + RNG.normal(size=N)
income    = 2.0 * education + 0.8 * ability + RNG.normal(size=N)

# ability is not collected — unobserved confounder
df = pd.DataFrame({"proximity": proximity, "education": education, "income": income})

dag = DAG()
dag.assume("proximity").causes("education")
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

print(dag)
print()

result = IV2SLS(dag, treatment="education", outcome="income", instrument="proximity").fit(df)
print(result.summary())
