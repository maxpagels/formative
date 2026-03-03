import numpy as np
import pandas as pd
import pytest

from formative import DAG, RCT
from formative.refutations.rct import RCTRefutationReport


RNG = np.random.default_rng(42)
N = 2_000
TRUE_ATE = 2.0


def make_dag():
    dag = DAG()
    dag.assume("treatment").causes("outcome")
    return dag


def make_data(true_ate=TRUE_ATE):
    treatment = RNG.integers(0, 2, size=N).astype(float)
    outcome = true_ate * treatment + RNG.normal(size=N)
    return pd.DataFrame({"treatment": treatment, "outcome": outcome})


class TestRCTRefutation:
    def test_random_common_cause_passes(self):
        df = make_data()
        result = RCT(make_dag(), treatment="treatment", outcome="outcome").fit(df)
        report = result.refute(df)
        assert report.passed
        assert len(report.checks) == 1
        assert report.checks[0].name == "Random common cause"
        assert report.checks[0].passed

    def test_report_type(self):
        df = make_data()
        result = RCT(make_dag(), treatment="treatment", outcome="outcome").fit(df)
        report = result.refute(df)
        assert isinstance(report, RCTRefutationReport)

    def test_summary_pass_labels(self):
        df = make_data()
        result = RCT(make_dag(), treatment="treatment", outcome="outcome").fit(df)
        summary = result.refute(df).summary()
        assert "PASS" in summary
        assert "All checks passed." in summary

    def test_passed_consistent_with_failed_checks(self):
        df = make_data()
        result = RCT(make_dag(), treatment="treatment", outcome="outcome").fit(df)
        report = result.refute(df)
        assert report.passed == (len(report.failed_checks) == 0)

    def test_checks_returns_copy(self):
        df = make_data()
        result = RCT(make_dag(), treatment="treatment", outcome="outcome").fit(df)
        report = result.refute(df)
        report.checks.clear()
        assert len(report.checks) == 1

    def test_summary_contains_treatment_and_outcome(self):
        df = make_data()
        result = RCT(make_dag(), treatment="treatment", outcome="outcome").fit(df)
        summary = result.refute(df).summary()
        assert "treatment" in summary
        assert "outcome" in summary

    def test_check_detail_populated(self):
        df = make_data()
        result = RCT(make_dag(), treatment="treatment", outcome="outcome").fit(df)
        report = result.refute(df)
        assert report.checks[0].detail != ""
