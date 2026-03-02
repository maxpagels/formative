"""
OLS with multiple confounders and a mediator.

DAG:
    parental_income → education → income
    parental_income ↗            ↗
                   ──────────────
    region         → education
    region         → income
    education      → job_type → income   (job_type is a mediator)

parental_income and region are both confounders: they affect both
education and income. Both are in the dataframe, so both are
controlled for automatically.

job_type is a mediator (a descendant of education). It is in the
dataframe but must NOT be controlled for — doing so would block part
of the causal path and underestimate the total effect of education.
OLSObservational correctly excludes it from the adjustment set.

True total effect of education on income: 1.68  (direct 1.2 + indirect 0.8×0.6)
"""

import numpy as np
import pandas as pd

from formative import DAG, OLSObservational

RNG = np.random.default_rng(0)
N = 2_000

parental_income = RNG.normal(size=N)
region          = RNG.integers(0, 3, size=N).astype(float)
education       = 0.4 * parental_income + 0.3 * region + RNG.normal(size=N)
job_type        = 0.6 * education + RNG.normal(size=N)
income          = (
    1.2 * education
    + 0.8 * job_type
    + 0.5 * parental_income
    + 0.4 * region
    + RNG.normal(size=N)
)

df = pd.DataFrame({
    "parental_income": parental_income,
    "region":          region,
    "education":       education,
    "job_type":        job_type,
    "income":          income,
})

dag = DAG()
dag.assume("parental_income").causes("education", "income")
dag.assume("region").causes("education", "income")
dag.assume("education").causes("job_type", "income")
dag.assume("job_type").causes("income")

print(dag)
print()
print("True total effect of education on income: 1.68  (direct 1.2 + indirect 0.8×0.6)")
print()

result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
print(result.summary())
print(result.executive_summary())
print(f"job_type in adjustment set: {'job_type' in result.adjustment_set}  (expected: False)")
