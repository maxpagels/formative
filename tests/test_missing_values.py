import numpy as np
import pandas as pd
import pytest

from formative.causal import DAG, IV2SLS, RCT, RDD, DiD, OLSObservational, PropensityScoreMatching

RNG = np.random.default_rng(42)
N = 200


def make_data():
    x = RNG.normal(size=N)
    t = (x + RNG.normal(size=N) > 0).astype(int)
    return pd.DataFrame(
        {
            "x": x,
            "z": RNG.normal(size=N),
            "g": RNG.integers(0, 2, size=N),
            "t": t,
            "y": 2 * t + x + RNG.normal(size=N),
        }
    )


def observational_dag():
    dag = DAG()
    dag.assume("z").causes("t")
    dag.assume("x").causes("t", "y")
    dag.assume("t").causes("y")
    return dag


def rct_dag():
    dag = DAG()
    dag.assume("t").causes("y")
    dag.assume("g").causes("y")
    return dag


def did_dag():
    dag = DAG()
    dag.assume("g").causes("y")
    dag.assume("t").causes("y")
    return dag


def rdd_dag():
    dag = DAG()
    dag.assume("x").causes("t", "y")
    dag.assume("t").causes("y")
    return dag


# (estimator, column to blank out) — the column is one the estimator actually uses.
CASES = [
    (lambda: OLSObservational(observational_dag(), treatment="t", outcome="y"), "x"),
    (lambda: OLSObservational(rct_dag(), treatment="t", outcome="y", effect_modifier="g"), "g"),
    (lambda: IV2SLS(observational_dag(), treatment="t", outcome="y", instrument="z"), "x"),
    (lambda: PropensityScoreMatching(observational_dag(), treatment="t", outcome="y"), "x"),
    (lambda: PropensityScoreMatching(observational_dag(), treatment="t", outcome="y"), "y"),
    (lambda: RCT(rct_dag(), treatment="t", outcome="y"), "y"),
    (lambda: RCT(rct_dag(), treatment="t", outcome="y", effect_modifier="g"), "g"),
    (lambda: DiD(did_dag(), group="g", time="t", outcome="y"), "t"),
    (lambda: RDD(rdd_dag(), treatment="t", running_var="x", cutoff=0.0, outcome="y"), "x"),
]


@pytest.mark.parametrize("make_estimator, column", CASES)
def test_missing_values_raise(make_estimator, column):
    df = make_data().astype({column: float})
    df.loc[df.index[:10], column] = np.nan
    with pytest.raises(ValueError, match=rf"\['{column}'\] contain missing values"):
        make_estimator().fit(df)


def test_rdd_ignores_missing_values_outside_bandwidth():
    df = make_data()
    df.loc[df["x"].abs() > 1.0, "y"] = np.nan
    result = RDD(rdd_dag(), treatment="t", running_var="x", cutoff=0.0, outcome="y", bandwidth=1.0).fit(df)
    assert np.isfinite(result.effect)


def test_missing_values_in_unused_column_are_ignored():
    df = make_data().assign(unrelated=np.nan)
    result = OLSObservational(observational_dag(), treatment="t", outcome="y").fit(df)
    assert result._n == N
