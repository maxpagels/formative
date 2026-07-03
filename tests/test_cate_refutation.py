import numpy as np
import pandas as pd

from formative.causal import DAG, RCT, OLSObservational
from formative.causal.refutations.cate import (
    _check_placebo_modifier,
    _check_random_common_cause,
    _check_random_modifier,
)

RNG = np.random.default_rng(42)
N = 2_000


def make_dag():
    dag = DAG()
    dag.assume("ability").causes("education", "income")
    dag.assume("segment").causes("income")
    dag.assume("education").causes("income")
    return dag


def make_data():
    segment = RNG.integers(0, 3, size=N)
    ability = RNG.normal(size=N)
    education = ability + RNG.normal(size=N)
    income = (1.0 + segment) * education + 1.5 * ability + 0.5 * segment + RNG.normal(size=N)
    return pd.DataFrame({"segment": segment, "ability": ability, "education": education, "income": income})


class TestOLSCATERefutation:
    @classmethod
    def setup_class(cls):
        cls.df = make_data()
        cls.result = OLSObservational(
            make_dag(), treatment="education", outcome="income", effect_modifier="segment"
        ).fit(cls.df)
        cls.report = cls.result.refute(cls.df)

    def test_all_checks_pass_on_well_specified_data(self):
        assert self.report.passed
        assert [c.name for c in self.report.checks] == [
            "Random common cause",
            "Placebo modifier",
            "Random modifier",
        ]

    def test_summary_pass_labels(self):
        summary = self.report.summary()
        assert "PASS" in summary
        assert "All checks passed." in summary


class TestRCTCATERefutation:
    def test_all_checks_pass_on_well_specified_data(self):
        segment = RNG.integers(0, 3, size=N)
        treatment = RNG.integers(0, 2, size=N).astype(float)
        outcome = (1.0 + segment) * treatment + 0.3 * segment + RNG.normal(size=N)
        df = pd.DataFrame({"segment": segment, "treatment": treatment, "outcome": outcome})

        dag = DAG()
        dag.assume("treatment").causes("outcome")
        dag.assume("segment").causes("outcome")
        result = RCT(dag, treatment="treatment", outcome="outcome", effect_modifier="segment").fit(df)
        assert result.refute(df).passed


class TestCheckFailureBranches:
    """Exercise the FAIL and error-handling paths of the individual checks."""

    @classmethod
    def setup_class(cls):
        cls.df = make_data()

    def test_random_common_cause_fails_on_shifted_estimate(self):
        """A wildly wrong original effect must trip the 1-SE shift threshold."""
        check = _check_random_common_cause(
            self.df, "education", "income", "segment", {"ability"}, original_effect=999.0, original_se=0.1
        )
        assert not check.passed
        assert "destabilised" in check.detail

    def test_placebo_modifier_reports_fit_failure(self):
        """A constant modifier makes the interaction model unfittable; the check must fail, not raise."""
        df = self.df.assign(segment=1)
        check = _check_placebo_modifier(df, "education", "income", "segment", {"ability"})
        assert not check.passed
        assert "failed" in check.detail

    def test_random_modifier_reports_fit_failure(self):
        """A non-numeric treatment makes the model unfittable; the check must fail, not raise."""
        df = self.df.assign(education=self.df["education"].astype(str))
        check = _check_random_modifier(df, "education", "income", "segment", {"ability"})
        assert not check.passed
        assert "failed" in check.detail

    def test_placebo_modifier_fails_on_surviving_heterogeneity(self):
        """A degenerate outcome yields significant placebo heterogeneity; the check must flag it."""
        df = self.df.assign(income="not numeric")
        check = _check_placebo_modifier(df, "education", "income", "segment", {"ability"})
        assert not check.passed
        assert "spurious" in check.detail

    def test_random_modifier_fails_on_spurious_heterogeneity(self):
        """A degenerate outcome yields significant noise-modifier heterogeneity; the check must flag it."""
        df = self.df.assign(income="not numeric")
        check = _check_random_modifier(df, "education", "income", "segment", {"ability"})
        assert not check.passed
        assert "not trustworthy" in check.detail
