import numpy as np
import pandas as pd

from formative import DAG, RDD
from formative.refutations._check import RefutationCheck
from formative.refutations.rdd import RDDRefutationReport

N = 2_000
TRUE_LATE = 2.0
CUTOFF = 0.0


def make_dag():
    dag = DAG()
    dag.assume("score").causes("treatment", "outcome")
    dag.assume("treatment").causes("outcome")
    return dag


def make_data(true_late=TRUE_LATE):
    rng = np.random.default_rng(42)
    score = rng.uniform(-1, 1, size=N)
    outcome = 1.0 * score + true_late * (score >= CUTOFF).astype(float) + rng.normal(scale=0.3, size=N)
    return pd.DataFrame({"score": score, "outcome": outcome})


class TestRDDRefutationReport:
    """Fit and refute once per class; tests inspect the pre-computed report."""

    @classmethod
    def setup_class(cls):
        cls.df = make_data()
        cls.result = RDD(make_dag(), treatment="treatment", running_var="score", cutoff=CUTOFF, outcome="outcome").fit(
            cls.df
        )
        cls.report = cls.result.refute(cls.df)

    def test_refute_returns_report(self):
        assert isinstance(self.report, RDDRefutationReport)

    def test_report_has_two_checks(self):
        assert len(self.report.checks) == 2

    def test_check_names(self):
        names = {c.name for c in self.report.checks}
        assert "Placebo cutoff" in names
        assert "Random common cause" in names

    def test_all_checks_pass(self):
        assert self.report.passed
        assert all(c.passed for c in self.report.checks)

    def test_placebo_cutoff_detail(self):
        check = next(c for c in self.report.checks if c.name == "Placebo cutoff")
        assert "placebo estimate" in check.detail

    def test_rcc_detail(self):
        check = next(c for c in self.report.checks if c.name == "Random common cause")
        assert "estimate shifted by" in check.detail

    def test_passed_consistent_with_failed_checks(self):
        assert self.report.passed == (len(self.report.failed_checks) == 0)

    def test_checks_returns_copy(self):
        copy = self.report.checks
        copy.clear()
        assert len(self.report.checks) == 2

    def test_summary_pass_labels(self):
        summary = self.report.summary()
        assert "PASS" in summary
        assert "All checks passed." in summary

    def test_summary_contains_variables(self):
        summary = self.report.summary()
        assert "score" in summary
        assert "treatment" in summary
        assert "outcome" in summary

    def test_checks_are_refutation_check_instances(self):
        for check in self.report.checks:
            assert isinstance(check, RefutationCheck)
            assert isinstance(check.name, str)
            assert isinstance(check.passed, bool)
            assert isinstance(check.detail, str)
            assert check.detail != ""


class TestRDDRefutationFail:
    """Verify the placebo cutoff check fails when there is a real jump in the control region."""

    @classmethod
    def setup_class(cls):
        # score is evenly spaced so median of score < 0 is exactly -0.5
        rng = np.random.default_rng(42)
        score = np.linspace(-1, 1, N)
        # True jump at 0 (LATE = 2.0) plus a large spurious jump at -0.5
        outcome = (
            1.0 * score
            + 2.0 * (score >= 0).astype(float)
            + 10.0 * (score >= -0.5).astype(float)
            + rng.normal(scale=0.1, size=N)
        )
        cls.df = pd.DataFrame({"score": score, "outcome": outcome})
        cls.result = RDD(make_dag(), treatment="treatment", running_var="score", cutoff=CUTOFF, outcome="outcome").fit(
            cls.df
        )
        cls.report = cls.result.refute(cls.df)

    def test_placebo_cutoff_fails(self):
        placebo_check = next(c for c in self.report.checks if c.name == "Placebo cutoff")
        assert not placebo_check.passed

    def test_summary_fail_labels(self):
        summary = self.report.summary()
        assert "FAIL" in summary
        assert "check(s) failed" in summary
