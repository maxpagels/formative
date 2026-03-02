import numpy as np
import pandas as pd
import pytest

from formative import DAG, RCT


RNG = np.random.default_rng(42)
N = 2_000
TRUE_ATE = 2.0


def make_data(true_ate=TRUE_ATE):
    """
    Ground truth DGP for a clean RCT:
      treatment ~ Bernoulli(0.5)   [randomised]
      outcome   = true_ate * treatment + noise
    No confounders by construction.
    """
    treatment = RNG.integers(0, 2, size=N).astype(float)
    outcome = true_ate * treatment + RNG.normal(size=N)
    return pd.DataFrame({"treatment": treatment, "outcome": outcome})


class TestRCT:
    def test_recovers_true_ate(self):
        dag = DAG()
        dag.assume("treatment").causes("outcome")

        df = make_data(true_ate=TRUE_ATE)
        result = RCT(dag, treatment="treatment", outcome="outcome").fit(df)

        assert abs(result.effect - TRUE_ATE) < 0.1

    def test_result_attributes(self):
        dag = DAG()
        dag.assume("treatment").causes("outcome")

        df = make_data()
        result = RCT(dag, treatment="treatment", outcome="outcome").fit(df)

        lo, hi = result.conf_int
        assert lo < result.effect < hi
        assert result.std_err > 0
        assert 0 <= result.pvalue <= 1
        assert result.statsmodels_result is not None

    def test_treatment_with_parents_raises(self):
        """Declaring a cause of treatment contradicts random assignment."""
        dag = DAG()
        dag.assume("ability").causes("treatment", "outcome")
        dag.assume("treatment").causes("outcome")

        with pytest.raises(ValueError, match="randomly assigned"):
            RCT(dag, treatment="treatment", outcome="outcome")

    def test_unknown_treatment_raises(self):
        dag = DAG()
        dag.assume("A").causes("B")
        with pytest.raises(ValueError, match="Treatment"):
            RCT(dag, treatment="X", outcome="B")

    def test_missing_treatment_column_raises(self):
        dag = DAG()
        dag.assume("treatment").causes("outcome")

        df = make_data().drop(columns=["treatment"])
        with pytest.raises(ValueError, match="Treatment column"):
            RCT(dag, treatment="treatment", outcome="outcome").fit(df)

    def test_missing_outcome_column_raises(self):
        dag = DAG()
        dag.assume("treatment").causes("outcome")

        df = make_data().drop(columns=["outcome"])
        with pytest.raises(ValueError, match="Outcome column"):
            RCT(dag, treatment="treatment", outcome="outcome").fit(df)

    def test_assumptions_present(self):
        dag = DAG()
        dag.assume("treatment").causes("outcome")

        df = make_data()
        result = RCT(dag, treatment="treatment", outcome="outcome").fit(df)

        names = [a.name for a in result.assumptions]
        assert any("SUTVA" in n for n in names)
        assert any("Random assignment" in n for n in names)

    def test_summary_and_executive_summary_run(self):
        dag = DAG()
        dag.assume("treatment").causes("outcome")

        df = make_data()
        result = RCT(dag, treatment="treatment", outcome="outcome").fit(df)

        assert "RCT" in result.summary()
        assert "Executive Summary" in result.executive_summary()

    def test_refute_runs(self):
        dag = DAG()
        dag.assume("treatment").causes("outcome")

        df = make_data()
        result = RCT(dag, treatment="treatment", outcome="outcome").fit(df)
        report = result.refute(df)

        assert report.passed
        assert len(report.checks) == 1
