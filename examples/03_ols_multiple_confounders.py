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

True total effect of education on income: 2.0
"""

import numpy as np
import pandas as pd

from formative import DAG, OLSObservational

RNG = np.random.default_rng(0)
N = 2_000

parental_income = RNG.normal(size=N)
region          = RNG.integers(0, 3, size=N).astype(float)  # 0, 1, 2
education       = 0.4 * parental_income + 0.3 * region + RNG.normal(size=N)
job_type        = 0.6 * education + RNG.normal(size=N)       # mediator
income          = (
    1.2 * education           # direct effect
    + 0.8 * job_type          # indirect effect via job_type (total = 1.2 + 0.8*0.6 = 1.68... wait)
    + 0.5 * parental_income
    + 0.4 * region
    + RNG.normal(size=N)
)
# True total effect: direct (1.2) + indirect via job_type (0.8 * 0.6) = 1.68

df = pd.DataFrame({
    "parental_income": parental_income,
    "region":          region,
    "education":       education,
    "job_type":        job_type,
    "income":          income,
})

dag = DAG()
dag.causes("parental_income", "education")
dag.causes("parental_income", "income")
dag.causes("region",          "education")
dag.causes("region",          "income")
dag.causes("education",       "job_type")
dag.causes("job_type",        "income")
dag.causes("education",       "income")

print(dag)
print()
print("True total effect of education on income: 1.68  (direct 1.2 + indirect 0.8×0.6)")
print()

result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
print(result.summary())
print(f"job_type in adjustment set: {'job_type' in result.adjustment_set}  (expected: False)")
