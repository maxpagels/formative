"""
Basic RCT example: effect of a randomly assigned treatment on an outcome.

DAG:
    treatment → outcome

Treatment is randomly assigned (Bernoulli 0.5), so no confounder
adjustment is needed. The ATE is estimated directly as the difference
in mean outcomes between treated and control units.

The true causal effect is 2.0.
"""

import numpy as np
import pandas as pd

from formative import DAG, RCT

RNG = np.random.default_rng(0)
N = 3_000
TRUE_ATE = 3.0

treatment = RNG.integers(0, 2, size=N).astype(float)
outcome   = TRUE_ATE * treatment + RNG.normal(size=N)

df = pd.DataFrame({"treatment": treatment, "outcome": outcome})

dag = DAG()
dag.assume("treatment").causes("outcome")

print(dag)
print()

result = RCT(dag, treatment="treatment", outcome="outcome").fit(df)
print(result.summary())
print(result.executive_summary())
