import numpy as np
import pandas as pd

from formative.causal import DAG, DiD
from formative.causal.refutations._check import RefutationCheck
from formative.causal.refutations.did import DiDRefutationReport

N = 1_000
TRUE_ATT = 3.0


def make_dag():
    dag = DAG()
    dag.assume("group").causes("outcome")
    dag.assume("time").causes("outcome")
    return dag


def make_data(true_att=TRUE_ATT):
    """
    Standard DiD DGP satisfying parallel trends by construction:
      outcome = 2 + 1.5*group + 3*time + true_att*group*time + noise
    """
    rng = np.random.default_rng(42)
    group = rng.integers(0, 2, size=N).astype(float)
    time = rng.integers(0, 2, size=N).astype(float)
    outcome = 2.0 + 1.5 * group + 3.0 * time + true_att * group * time + rng.normal(size=N)
    return pd.DataFrame({"group": group, "time": time, "outcome": outcome})


class TestDiDRefutationReport:
    """Fit and refute once per class; tests inspect the pre-computed report."""

    @classmethod
    def setup_class(cls):
        cls.df = make_data()
        cls.result = DiD(make_dag(), group="group", time="time", outcome="outcome").fit(cls.df)
        cls.report = cls.result.refute(cls.df)

    def test_refute_returns_report(self):
        assert isinstance(self.report, DiDRefutationReport)

    def test_report_has_three_checks(self):
        assert len(self.report.checks) == 3

    def test_check_names(self):
        names = {c.name for c in self.report.checks}
        assert "Placebo group" in names
        assert "Placebo time" in names
        assert "Random common cause" in names

    def test_all_checks_pass(self):
        assert self.report.passed
        assert all(c.passed for c in self.report.checks)

    def test_placebo_group_detail(self):
        check = next(c for c in self.report.checks if c.name == "Placebo group")
        assert "placebo estimate" in check.detail
        assert "near-zero effect" in check.detail

    def test_placebo_time_detail(self):
        check = next(c for c in self.report.checks if c.name == "Placebo time")
        assert "placebo estimate" in check.detail
        assert "near-zero effect" in check.detail

    def test_rcc_detail(self):
        check = next(c for c in self.report.checks if c.name == "Random common cause")
        assert "estimate shifted by" in check.detail

    def test_passed_consistent_with_failed_checks(self):
        assert self.report.passed == (len(self.report.failed_checks) == 0)

    def test_checks_returns_copy(self):
        copy = self.report.checks
        copy.clear()
        assert len(self.report.checks) == 3

    def test_summary_pass_labels(self):
        summary = self.report.summary()
        assert "PASS" in summary
        assert "All checks passed." in summary

    def test_summary_contains_variables(self):
        summary = self.report.summary()
        assert "group" in summary
        assert "time" in summary
        assert "outcome" in summary

    def test_checks_are_refutation_check_instances(self):
        for check in self.report.checks:
            assert isinstance(check, RefutationCheck)
            assert isinstance(check.name, str)
            assert isinstance(check.passed, bool)
            assert isinstance(check.detail, str)
            assert check.detail != ""


class TestDiDRefutationFail:
    def test_placebo_group_fails_on_spurious_data(self):
        """
        When the outcome contains a group-time interaction that has nothing
        to do with treatment (i.e. parallel trends is violated by design),
        permuting group labels can still recover a large interaction,
        causing the placebo group check to fail.
        """
        rng = np.random.default_rng(7)
        N = 2_000
        group = rng.integers(0, 2, size=N).astype(float)
        time = rng.integers(0, 2, size=N).astype(float)
        # Large spurious confound correlated with group*time drives the interaction
        confound = 20.0 * group * time
        outcome = 2.0 + 1.5 * group + 3.0 * time + 3.0 * group * time + confound + rng.normal(size=N)
        df = pd.DataFrame({"group": group, "time": time, "outcome": outcome})

        dag = make_dag()
        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)
        report = result.refute(df)

        placebo = next(c for c in report.checks if c.name == "Placebo group")
        # The placebo effect here should be large, causing a failure
        assert not placebo.passed

    def test_summary_fail_labels(self):
        rng = np.random.default_rng(7)
        N = 2_000
        group = rng.integers(0, 2, size=N).astype(float)
        time = rng.integers(0, 2, size=N).astype(float)
        confound = 20.0 * group * time
        outcome = 2.0 + 1.5 * group + 3.0 * time + 3.0 * group * time + confound + rng.normal(size=N)
        df = pd.DataFrame({"group": group, "time": time, "outcome": outcome})

        dag = make_dag()
        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)
        summary = result.refute(df).summary()

        assert "FAIL" in summary
        assert "check(s) failed" in summary
