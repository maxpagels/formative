"""
Decision analysis example: should we run a job training programme?

DAG:
    background → training → earnings
    background ↗            ↗
              ──────────────

background is a confounder: family/educational background affects both
who selects into training and their subsequent earnings.
OLSObservational controls for it automatically.

The true causal effect of training on earnings is 3.0 units.

Decision question:
    The programme costs $8 per participant.
    Each unit of earnings increase is worth $15 in lifetime value.
    Should we roll it out?
"""

import numpy as np
import pandas as pd

from formative import DAG, OLSObservational

RNG = np.random.default_rng(0)
N = 2_000

background = RNG.normal(size=N)
training = 0.6 * background + RNG.normal(size=N)
earnings = 3.0 * training + 1.2 * background + RNG.normal(size=N)

df = pd.DataFrame({"background": background, "training": training, "earnings": earnings})

dag = DAG()
dag.assume("background").causes("training", "earnings")
dag.assume("training").causes("earnings")

result = OLSObservational(dag, treatment="training", outcome="earnings").fit(df)
print(result.summary())

# ── Decision analysis ──────────────────────────────────────────────────────────
# cost    = $8 per participant
# benefit = $15 per unit of earnings increase (lifetime value)

decision = result.decide(cost=8, benefit=15)
print(decision)

# How much more data would we need if we were less certain?
print(decision.value_of_information(target_confidence=0.99))
