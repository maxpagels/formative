"""
IV with an observed confounder included as a control.

When some confounders are observed and others are not, IV2SLS automatically
includes the observed ones as controls in both stages (improving efficiency),
while the instrument handles the unobserved confounding.

DAG:
    proximity  → education → income
    socioeconomic_status → education
    socioeconomic_status → income
    ability (unobserved) → education
    ability (unobserved) → income

socioeconomic_status is observed and added to the adjustment set automatically.
ability is unobserved and handled by the instrument.

True effect of education on income: 2.0
"""

import numpy as np
import pandas as pd

from formative import DAG, IV2SLS

RNG = np.random.default_rng(0)
N = 5_000

proximity            = RNG.normal(size=N)
ability              = RNG.normal(size=N)   # unobserved
socioeconomic_status = RNG.normal(size=N)   # observed confounder
education = (
    0.5 * proximity
    + 0.4 * ability
    + 0.3 * socioeconomic_status
    + RNG.normal(size=N)
)
income = (
    2.0 * education
    + 0.8 * ability
    + 0.5 * socioeconomic_status
    + RNG.normal(size=N)
)

# ability is not collected
df = pd.DataFrame({
    "proximity":            proximity,
    "socioeconomic_status": socioeconomic_status,
    "education":            education,
    "income":               income,
})

dag = DAG()
dag.assume("proximity").causes("education")
dag.assume("ability").causes("education", "income")
dag.assume("socioeconomic_status").causes("education", "income")
dag.assume("education").causes("income")

result = IV2SLS(
    dag, treatment="education", outcome="income", instrument="proximity"
).fit(df)

print(result.summary())
print(result.executive_summary())
print(f"adjustment_set: {result.adjustment_set}  (socioeconomic_status controlled, ability handled by IV)")
