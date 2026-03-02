import numpy as np
import pandas as pd
import pytest

from formative import DAG, DiD, DiDResult


RNG = np.random.default_rng(42)
N = 1_000
TRUE_ATT = 3.0


def make_data(true_att=TRUE_ATT):
    """
    Standard DiD DGP satisfying parallel trends by construction:
      group  ~ Bernoulli(0.5)         [treatment group indicator]
      time   ~ Bernoulli(0.5)         [post-period indicator]
      outcome = 2 + 1.5*group + 3*time + true_att*group*time + noise
    The true ATT is the coefficient on group*time.
    """
    group = RNG.integers(0, 2, size=N).astype(float)
    time = RNG.integers(0, 2, size=N).astype(float)
    outcome = (
        2.0
        + 1.5 * group
        + 3.0 * time
        + true_att * group * time
        + RNG.normal(size=N)
    )
    return pd.DataFrame({"group": group, "time": time, "outcome": outcome})


class TestDiD:
    def test_recovers_true_att(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data(true_att=TRUE_ATT)
        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)

        assert abs(result.effect - TRUE_ATT) < 0.2

    def test_result_attributes(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data()
        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)

        lo, hi = result.conf_int
        assert lo < result.effect < hi
        assert result.std_err > 0
        assert 0 <= result.pvalue <= 1
        assert result.statsmodels_result is not None

    def test_naive_diff(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data()
        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)

        # Naive diff is treated post mean minus control post mean
        treated_post = df.loc[(df["group"] == 1) & (df["time"] == 1), "outcome"].mean()
        control_post = df.loc[(df["group"] == 0) & (df["time"] == 1), "outcome"].mean()
        expected_naive = treated_post - control_post

        assert abs(result.naive_diff - expected_naive) < 1e-10

    def test_group_not_in_dag_raises(self):
        dag = DAG()
        dag.assume("time").causes("outcome")

        with pytest.raises(ValueError, match="Group"):
            DiD(dag, group="group", time="time", outcome="outcome")

    def test_time_not_in_dag_raises(self):
        dag = DAG()
        dag.assume("group").causes("outcome")

        with pytest.raises(ValueError, match="Time"):
            DiD(dag, group="group", time="time", outcome="outcome")

    def test_outcome_not_in_dag_raises(self):
        dag = DAG()
        dag.assume("group").causes("time")

        with pytest.raises(ValueError, match="Outcome"):
            DiD(dag, group="group", time="time", outcome="outcome")

    def test_overlapping_variables_raises(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        with pytest.raises(ValueError, match="different variables"):
            DiD(dag, group="group", time="group", outcome="outcome")

    def test_non_binary_group_raises(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data()
        df["group"] = df["group"] * 2  # now 0/2 instead of 0/1

        with pytest.raises(ValueError, match="binary"):
            DiD(dag, group="group", time="time", outcome="outcome").fit(df)

    def test_non_binary_time_raises(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data()
        df["time"] = df["time"].replace({0: 0, 1: 2})

        with pytest.raises(ValueError, match="binary"):
            DiD(dag, group="group", time="time", outcome="outcome").fit(df)

    def test_missing_group_column_raises(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data().drop(columns=["group"])
        with pytest.raises(ValueError, match="Group column"):
            DiD(dag, group="group", time="time", outcome="outcome").fit(df)

    def test_missing_outcome_column_raises(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data().drop(columns=["outcome"])
        with pytest.raises(ValueError, match="Outcome column"):
            DiD(dag, group="group", time="time", outcome="outcome").fit(df)

    def test_assumptions_present(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data()
        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)

        names = [a.name for a in result.assumptions]
        assert any("arallel trends" in n for n in names)
        assert any("SUTVA" in n for n in names)

    def test_summary_and_executive_summary_run(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data()
        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)

        assert "DiD" in result.summary()
        assert "Executive Summary" in result.executive_summary()

    def test_refute_runs(self):
        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        df = make_data()
        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)
        report = result.refute(df)

        assert report.passed
        assert len(report.checks) == 3
