import numpy as np
import pandas as pd
import pytest

from formative import DAG, IV2SLS
from formative.refutations._check import RefutationCheck
from formative.refutations.iv import IVRefutationReport


RNG = np.random.default_rng(42)
N = 5_000


def make_dag():
    dag = DAG()
    dag.assume("proximity").causes("education")
    dag.assume("ability").causes("education", "income")
    dag.assume("education").causes("income")
    return dag


def make_strong_instrument_data():
    """proximity has a strong effect on education (F >> 10)."""
    proximity = RNG.normal(size=N)
    ability   = RNG.normal(size=N)
    education = 0.5 * proximity + 0.5 * ability + RNG.normal(size=N)
    income    = 2.0 * education + 0.8 * ability + RNG.normal(size=N)
    return pd.DataFrame({"proximity": proximity, "education": education, "income": income})


def make_weak_instrument_data():
    """proximity barely moves education (F < 10)."""
    proximity = RNG.normal(size=N)
    ability   = RNG.normal(size=N)
    education = 0.02 * proximity + 0.5 * ability + RNG.normal(size=N)
    income    = 2.0 * education + 0.8 * ability + RNG.normal(size=N)
    return pd.DataFrame({"proximity": proximity, "education": education, "income": income})


class TestIVRefutationReport:
    def test_strong_instrument_passes(self):
        df = make_strong_instrument_data()
        result = IV2SLS(make_dag(), treatment="education", outcome="income", instrument="proximity").fit(df)
        report = result.refute(df)
        assert report.passed
        assert all(c.passed for c in report.checks)

    def test_weak_instrument_fails(self):
        df = make_weak_instrument_data()
        result = IV2SLS(make_dag(), treatment="education", outcome="income", instrument="proximity").fit(df)
        report = result.refute(df)
        assert not report.passed
        assert len(report.failed_checks) == 1
        assert "Weak instrument" in report.failed_checks[0].detail

    def test_first_stage_check_name(self):
        df = make_strong_instrument_data()
        result = IV2SLS(make_dag(), treatment="education", outcome="income", instrument="proximity").fit(df)
        report = result.refute(df)
        assert report.checks[0].name == "First-stage F-statistic"

    def test_summary_pass_labels(self):
        df = make_strong_instrument_data()
        result = IV2SLS(make_dag(), treatment="education", outcome="income", instrument="proximity").fit(df)
        summary = result.refute(df).summary()
        assert "PASS" in summary
        assert "All checks passed." in summary

    def test_summary_fail_labels(self):
        df = make_weak_instrument_data()
        result = IV2SLS(make_dag(), treatment="education", outcome="income", instrument="proximity").fit(df)
        summary = result.refute(df).summary()
        assert "FAIL" in summary
        assert "check(s) failed" in summary

    def test_passed_property_consistent_with_failed_checks(self):
        for df in [make_strong_instrument_data(), make_weak_instrument_data()]:
            result = IV2SLS(make_dag(), treatment="education", outcome="income", instrument="proximity").fit(df)
            report = result.refute(df)
            assert report.passed == (len(report.failed_checks) == 0)

    def test_checks_returns_copy(self):
        df = make_strong_instrument_data()
        result = IV2SLS(make_dag(), treatment="education", outcome="income", instrument="proximity").fit(df)
        report = result.refute(df)
        report.checks.clear()
        assert len(report.checks) == 2

    def test_random_common_cause_passes(self):
        df = make_strong_instrument_data()
        result = IV2SLS(make_dag(), treatment="education", outcome="income", instrument="proximity").fit(df)
        report = result.refute(df)
        rcc = next(c for c in report.checks if c.name == "Random common cause")
        assert rcc.passed

    def test_report_has_two_checks(self):
        df = make_strong_instrument_data()
        result = IV2SLS(make_dag(), treatment="education", outcome="income", instrument="proximity").fit(df)
        report = result.refute(df)
        assert len(report.checks) == 2
        names = {c.name for c in report.checks}
        assert "First-stage F-statistic" in names
        assert "Random common cause" in names
