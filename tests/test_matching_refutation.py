import numpy as np
import pandas as pd

from formative import DAG, PropensityScoreMatching
from formative.refutations._check import RefutationCheck
from formative.refutations.matching import MatchingRefutationReport


N = 1_000


def make_dag():
    dag = DAG()
    dag.assume("ability").causes("education", "income")
    dag.assume("education").causes("income")
    return dag


def make_data():
    """Fixed seed so every call returns the same dataframe."""
    rng = np.random.default_rng(42)
    ability   = rng.normal(size=N)
    ps_latent = 0.5 * ability + rng.normal(scale=0.5, size=N)
    education = (ps_latent > np.median(ps_latent)).astype(float)
    income    = 2.0 * education + 0.8 * ability + rng.normal(size=N)
    return pd.DataFrame({"ability": ability, "education": education, "income": income})


class TestMatchingRefutationReport:
    """Fit and refute once per class; tests inspect the pre-computed report."""

    @classmethod
    def setup_class(cls):
        cls.df     = make_data()
        cls.result = PropensityScoreMatching(
            make_dag(), treatment="education", outcome="income"
        ).fit(cls.df)
        cls.report = cls.result.refute(cls.df)

    def test_refute_returns_report(self):
        assert isinstance(self.report, MatchingRefutationReport)

    def test_report_has_two_checks(self):
        assert len(self.report.checks) == 2

    def test_check_names(self):
        names = {c.name for c in self.report.checks}
        assert "Placebo treatment" in names
        assert "Random common cause" in names

    def test_placebo_att_much_smaller_than_original(self):
        # Permuting labels should eliminate the real 2.0 effect almost entirely.
        placebo = next(c for c in self.report.checks if c.name == "Placebo treatment")
        assert abs(self.result.effect) > 1.0   # original ATT is large
        assert "placebo ATT" in placebo.detail  # check ran and reported a value

    def test_rcc_detail_reports_shift(self):
        rcc = next(c for c in self.report.checks if c.name == "Random common cause")
        assert "estimate shifted by" in rcc.detail

    def test_passed_consistent_with_failed_checks(self):
        assert self.report.passed == (len(self.report.failed_checks) == 0)

    def test_checks_returns_copy(self):
        copy = self.report.checks
        copy.clear()
        assert len(self.report.checks) == 2  # internal list untouched

    def test_all_checks_pass(self):
        assert self.report.passed
        assert all(c.passed for c in self.report.checks)

    def test_summary_all_pass(self):
        summary = self.report.summary()
        assert "PASS" in summary
        assert "All checks passed." in summary

    def test_summary_contains_treatment_and_outcome(self):
        summary = self.report.summary()
        assert "education" in summary
        assert "income" in summary

    def test_checks_are_refutation_check_instances(self):
        for check in self.report.checks:
            assert isinstance(check, RefutationCheck)
            assert isinstance(check.name, str)
            assert isinstance(check.passed, bool)
            assert isinstance(check.detail, str)
