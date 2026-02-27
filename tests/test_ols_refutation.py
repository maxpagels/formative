import numpy as np
import pandas as pd
import pytest

from formative import DAG, OLSObservational


RNG = np.random.default_rng(42)
N = 2_000


def make_dag():
    dag = DAG()
    dag.assume("ability").causes("education", "income")
    dag.assume("education").causes("income")
    return dag


def make_data(true_effect=2.0):
    ability   = RNG.normal(size=N)
    education = 0.5 * ability + RNG.normal(size=N)
    income    = true_effect * education + 0.8 * ability + RNG.normal(size=N)
    return pd.DataFrame({"ability": ability, "education": education, "income": income})


class TestOLSRefutation:
    def test_random_common_cause_passes(self):
        df = make_data()
        result = OLSObservational(make_dag(), treatment="education", outcome="income").fit(df)
        report = result.refute(df)
        assert report.passed
        assert len(report.checks) == 1
        assert report.checks[0].name == "Random common cause"
        assert report.checks[0].passed

    def test_summary_pass_labels(self):
        df = make_data()
        result = OLSObservational(make_dag(), treatment="education", outcome="income").fit(df)
        summary = result.refute(df).summary()
        assert "PASS" in summary
        assert "All checks passed." in summary

    def test_passed_consistent_with_failed_checks(self):
        df = make_data()
        result = OLSObservational(make_dag(), treatment="education", outcome="income").fit(df)
        report = result.refute(df)
        assert report.passed == (len(report.failed_checks) == 0)

    def test_checks_returns_copy(self):
        df = make_data()
        result = OLSObservational(make_dag(), treatment="education", outcome="income").fit(df)
        report = result.refute(df)
        report.checks.clear()
        assert len(report.checks) == 1
