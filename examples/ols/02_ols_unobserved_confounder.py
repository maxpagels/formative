"""
What happens when a confounder is not in the dataframe.

Same setup as 01_ols_basic.py, but ability is not collected.
OLSObservational detects that ability is a confounder in the DAG
but absent from the data, and raises an IdentificationError.

Compare the naive OLS estimate (biased) with the error message.
"""

import numpy as np
import pandas as pd

from formative import DAG, OLSObservational
from formative._exceptions import IdentificationError

RNG = np.random.default_rng(0)
N = 2_000

ability    = RNG.normal(size=N)
education  = 0.5 * ability + RNG.normal(size=N)
income     = 2.0 * education + 0.8 * ability + RNG.normal(size=N)

# ability is not included in the dataframe — we never collected it
df = pd.DataFrame({"education": education, "income": income})

dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

# For reference: naive OLS ignoring the confounder
import statsmodels.formula.api as smf
naive = smf.ols("income ~ education", data=df).fit()
print(f"Naive OLS estimate (biased): {naive.params['education']:.4f}  (true effect: 2.0)")
print()

# formative refuses to run
try:
    OLSObservational(dag, treatment="education", outcome="income").fit(df)
except IdentificationError as e:
    print("IdentificationError:", e)
