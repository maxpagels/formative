import numpy as np
import pandas as pd
import pytest

from formative.causal import DAG, IV2SLS, RCT, RDD, DiD, OLSObservational, PropensityScoreMatching

N = 200


def make_data(rename=None):
    rng = np.random.default_rng(42)
    x = rng.normal(size=N)
    t = (x + rng.normal(size=N) > 0).astype(int)
    df = pd.DataFrame(
        {
            "x": x,
            "z": rng.normal(size=N),
            "g": rng.integers(0, 2, size=N),
            "t": t,
            "y": 2 * t + x + rng.normal(size=N),
        }
    )
    return df.rename(columns=rename or {})


def dag(edges, rename=None):
    rename = rename or {}
    d = DAG()
    for cause, effects in edges:
        d.assume(rename.get(cause, cause)).causes(*(rename.get(e, e) for e in effects))
    return d


OBSERVATIONAL = [("z", ["t"]), ("x", ["t", "y"]), ("t", ["y"])]
RCT_EDGES = [("t", ["y"]), ("g", ["y"])]
DID_EDGES = [("g", ["y"]), ("t", ["y"])]
RDD_EDGES = [("x", ["t", "y"]), ("t", ["y"])]


BAD = "my col"

# Each factory takes a rename map so the DAG and data use the same (bad) name.
ESTIMATORS = {
    "ols": lambda r: OLSObservational(dag(OBSERVATIONAL, r), treatment=r.get("t", "t"), outcome=r.get("y", "y")),
    "ols_modifier": lambda r: OLSObservational(
        dag(RCT_EDGES, r), treatment=r.get("t", "t"), outcome=r.get("y", "y"), effect_modifier=r.get("g", "g")
    ),
    "iv": lambda r: IV2SLS(
        dag(OBSERVATIONAL, r), treatment=r.get("t", "t"), outcome=r.get("y", "y"), instrument=r.get("z", "z")
    ),
    "matching": lambda r: PropensityScoreMatching(
        dag(OBSERVATIONAL, r), treatment=r.get("t", "t"), outcome=r.get("y", "y")
    ),
    "rct": lambda r: RCT(dag(RCT_EDGES, r), treatment=r.get("t", "t"), outcome=r.get("y", "y")),
    "did": lambda r: DiD(dag(DID_EDGES, r), group=r.get("g", "g"), time=r.get("t", "t"), outcome=r.get("y", "y")),
    "rdd": lambda r: RDD(
        dag(RDD_EDGES, r), treatment=r.get("t", "t"), running_var=r.get("x", "x"), cutoff=0.0, outcome=r.get("y", "y")
    ),
}

# Columns each estimator puts into a statsmodels formula.
FORMULA_COLUMNS = [
    ("ols", "t"),
    ("ols", "x"),
    ("ols_modifier", "g"),
    ("iv", "y"),
    ("matching", "x"),
    ("rct", "y"),
    ("did", "g"),
    ("rdd", "y"),
]

# Columns that never enter a formula, so any name works.
NON_FORMULA_COLUMNS = [("iv", "z"), ("rdd", "x")]


@pytest.mark.parametrize("estimator, column", FORMULA_COLUMNS)
def test_invalid_formula_column_name_raises(estimator, column):
    rename = {column: BAD}
    with pytest.raises(ValueError, match=r"\['my col'\] can't be used in a model formula.*'my_col'"):
        ESTIMATORS[estimator](rename).fit(make_data(rename))


@pytest.mark.parametrize("estimator, column", NON_FORMULA_COLUMNS)
def test_non_formula_column_may_have_any_name(estimator, column):
    rename = {column: BAD}
    result = ESTIMATORS[estimator](rename).fit(make_data(rename))
    assert np.isfinite(result.effect)


@pytest.mark.parametrize("name", ["class", "2020_sales", "ad-spend"])
def test_keywords_and_non_identifiers_rejected(name):
    rename = {"y": name}
    with pytest.raises(ValueError, match="can't be used in a model formula"):
        ESTIMATORS["rct"](rename).fit(make_data(rename))
