"""
Basic DiD example: effect of a policy intervention on an outcome.

DAG:
    group   → outcome
    time    → outcome

A policy is introduced for the treated group (group = 1) in the post
period (time = 1). The control group (group = 0) is never treated.

Under parallel trends, DiD removes the baseline difference between
groups and the common time trend, isolating the treatment effect.

The true ATT is 4.0.
"""

import numpy as np
import pandas as pd

from formative import DAG, DiD

RNG = np.random.default_rng(42)
N = 1_000
TRUE_ATT = 4.0

group = RNG.integers(0, 2, size=N).astype(float)
time  = RNG.integers(0, 2, size=N).astype(float)

# DGP satisfies parallel trends by construction
outcome = (
    3.0                        # intercept
    + 2.0 * group              # baseline group difference
    + 1.5 * time               # common time trend
    + TRUE_ATT * group * time  # treatment effect (ATT)
    + RNG.normal(size=N)
)

df = pd.DataFrame({"group": group, "time": time, "outcome": outcome})

dag = DAG()
dag.assume("group").causes("outcome")
dag.assume("time").causes("outcome")

print(dag)
print()

result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)
print(result.summary())
print(result.executive_summary())
print(result.statsmodels_result.summary())

report = result.refute(df)
print(report.summary())