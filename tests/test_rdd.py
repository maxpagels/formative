import numpy as np
import pandas as pd
import pytest

from formative import DAG, RDD, RDDResult


N = 2_000
TRUE_LATE = 2.0
CUTOFF = 0.0


def make_dag():
    dag = DAG()
    dag.assume("score").causes("treatment", "outcome")
    dag.assume("treatment").causes("outcome")
    return dag


def make_data(true_late=TRUE_LATE):
    """
    Sharp RDD DGP:
      score   ~ Uniform(-1, 1)
      outcome = 1.0 * score + true_late * (score >= 0) + noise

    Treatment is derived by RDD.fit() — not included in the dataframe.
    The slope of score on outcome causes the naive mean diff to exceed the
    true LATE (upward bias corrected by local linear regression).
    """
    rng = np.random.default_rng(42)
    score = rng.uniform(-1, 1, size=N)
    outcome = (
        1.0 * score
        + true_late * (score >= CUTOFF).astype(float)
        + rng.normal(scale=0.3, size=N)
    )
    return pd.DataFrame({"score": score, "outcome": outcome})


class TestRDD:
    def test_recovers_true_late(self):
        df = make_data()
        result = RDD(make_dag(), treatment="treatment", running_var="score",
                     cutoff=CUTOFF, outcome="outcome").fit(df)
        assert abs(result.effect - TRUE_LATE) < 0.1

    def test_result_attributes(self):
        df = make_data()
        result = RDD(make_dag(), treatment="treatment", running_var="score",
                     cutoff=CUTOFF, outcome="outcome").fit(df)

        lo, hi = result.conf_int
        assert lo < result.effect < hi
        assert result.std_err > 0
        assert 0 <= result.pvalue <= 1
        assert result.statsmodels_result is not None
        assert result.n_obs == N
        assert result.cutoff == CUTOFF
        assert result.running_var == "score"
        assert result.bandwidth is None

    def test_unadjusted_is_biased(self):
        """Naive mean diff should exceed the true LATE because the slope of score inflates above-cutoff mean."""
        df = make_data()
        result = RDD(make_dag(), treatment="treatment", running_var="score",
                     cutoff=CUTOFF, outcome="outcome").fit(df)
        assert result.unadjusted_effect > result.effect

    def test_bandwidth_restricts_data(self):
        df = make_data()
        result_full = RDD(make_dag(), treatment="treatment", running_var="score",
                          cutoff=CUTOFF, outcome="outcome").fit(df)
        result_bw = RDD(make_dag(), treatment="treatment", running_var="score",
                        cutoff=CUTOFF, outcome="outcome", bandwidth=0.5).fit(df)
        assert result_bw.n_obs < result_full.n_obs

    def test_bandwidth_recovers_true_late(self):
        df = make_data()
        result = RDD(make_dag(), treatment="treatment", running_var="score",
                     cutoff=CUTOFF, outcome="outcome", bandwidth=0.5).fit(df)
        assert abs(result.effect - TRUE_LATE) < 0.15

    def test_running_var_not_in_dag_raises(self):
        dag = DAG()
        dag.assume("treatment").causes("outcome")
        with pytest.raises(ValueError, match="Running variable"):
            RDD(dag, treatment="treatment", running_var="score",
                cutoff=CUTOFF, outcome="outcome")

    def test_treatment_not_in_dag_raises(self):
        dag = DAG()
        dag.assume("score").causes("outcome")
        with pytest.raises(ValueError, match="Treatment"):
            RDD(dag, treatment="treatment", running_var="score",
                cutoff=CUTOFF, outcome="outcome")

    def test_outcome_not_in_dag_raises(self):
        dag = DAG()
        dag.assume("score").causes("treatment")
        with pytest.raises(ValueError, match="Outcome"):
            RDD(dag, treatment="treatment", running_var="score",
                cutoff=CUTOFF, outcome="outcome")

    def test_overlapping_variables_raises(self):
        dag = DAG()
        dag.assume("score").causes("treatment", "outcome")
        dag.assume("treatment").causes("outcome")
        with pytest.raises(ValueError, match="different variables"):
            RDD(dag, treatment="treatment", running_var="score",
                cutoff=CUTOFF, outcome="treatment")

    def test_running_var_not_ancestor_of_treatment_raises(self):
        # score is not an ancestor of treatment — no path from score to treatment
        dag = DAG()
        dag.assume("score").causes("outcome")
        dag.assume("treatment").causes("outcome")
        with pytest.raises(ValueError, match="ancestor"):
            RDD(dag, treatment="treatment", running_var="score",
                cutoff=CUTOFF, outcome="outcome")

    def test_missing_running_var_column_raises(self):
        df = make_data().drop(columns=["score"])
        with pytest.raises(ValueError, match="Running variable column"):
            RDD(make_dag(), treatment="treatment", running_var="score",
                cutoff=CUTOFF, outcome="outcome").fit(df)

    def test_missing_outcome_column_raises(self):
        df = make_data().drop(columns=["outcome"])
        with pytest.raises(ValueError, match="Outcome column"):
            RDD(make_dag(), treatment="treatment", running_var="score",
                cutoff=CUTOFF, outcome="outcome").fit(df)

    def test_assumptions_present(self):
        df = make_data()
        result = RDD(make_dag(), treatment="treatment", running_var="score",
                     cutoff=CUTOFF, outcome="outcome").fit(df)
        names = [a.name for a in result.assumptions]
        assert any("Continuity" in n for n in names)
        assert any("manipulation" in n for n in names)

    def test_summary_and_executive_summary_run(self):
        df = make_data()
        result = RDD(make_dag(), treatment="treatment", running_var="score",
                     cutoff=CUTOFF, outcome="outcome").fit(df)
        assert "RDD" in result.summary()
        assert "Executive Summary" in result.executive_summary()

    def test_refute_runs(self):
        df = make_data()
        result = RDD(make_dag(), treatment="treatment", running_var="score",
                     cutoff=CUTOFF, outcome="outcome").fit(df)
        report = result.refute(df)
        assert report.passed
        assert len(report.checks) == 2
