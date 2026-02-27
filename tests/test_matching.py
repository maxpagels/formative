import numpy as np
import pandas as pd
import pytest

from formative import DAG, PropensityScoreMatching, MatchingResult
from formative._exceptions import IdentificationError


N = 1_000


def make_dag():
    dag = DAG()
    dag.assume("ability").causes("education", "income")
    dag.assume("education").causes("income")
    return dag


def make_data(true_att=2.0):
    """Fixed seed so every call returns the same dataframe."""
    rng = np.random.default_rng(42)
    ability   = rng.normal(size=N)
    ps_latent = 0.5 * ability + rng.normal(scale=0.5, size=N)
    education = (ps_latent > np.median(ps_latent)).astype(float)
    income    = true_att * education + 0.8 * ability + rng.normal(size=N)
    return pd.DataFrame({"ability": ability, "education": education, "income": income})


class TestPropensityScoreMatchingValidation:
    """Input validation — all raise before the bootstrap loop, so no setup needed."""

    def test_treatment_not_in_dag_raises(self):
        with pytest.raises(ValueError, match="Treatment"):
            PropensityScoreMatching(make_dag(), treatment="wage", outcome="income")

    def test_outcome_not_in_dag_raises(self):
        with pytest.raises(ValueError, match="Outcome"):
            PropensityScoreMatching(make_dag(), treatment="education", outcome="wage")

    def test_treatment_equals_outcome_raises(self):
        with pytest.raises(ValueError, match="different"):
            PropensityScoreMatching(make_dag(), treatment="education", outcome="education")

    def test_missing_treatment_column_raises(self):
        df = make_data().drop(columns=["education"])
        with pytest.raises(ValueError, match="Treatment"):
            PropensityScoreMatching(make_dag(), treatment="education", outcome="income").fit(df)

    def test_missing_outcome_column_raises(self):
        df = make_data().drop(columns=["income"])
        with pytest.raises(ValueError, match="Outcome"):
            PropensityScoreMatching(make_dag(), treatment="education", outcome="income").fit(df)

    def test_non_binary_treatment_raises(self):
        df = make_data()
        df["education"] = df["education"] * 3 + 1  # values 1 and 4
        with pytest.raises(ValueError, match="binary"):
            PropensityScoreMatching(make_dag(), treatment="education", outcome="income").fit(df)

    def test_only_treated_raises(self):
        df = make_data()
        df["education"] = 1.0
        with pytest.raises(ValueError):
            PropensityScoreMatching(make_dag(), treatment="education", outcome="income").fit(df)

    def test_missing_confounder_raises_identification_error(self):
        df = make_data().drop(columns=["ability"])
        with pytest.raises(IdentificationError):
            PropensityScoreMatching(make_dag(), treatment="education", outcome="income").fit(df)


class TestPropensityScoreMatchingEstimation:
    """Fit once per class (setup_class) so the bootstrap runs only once."""

    @classmethod
    def setup_class(cls):
        cls.df     = make_data(true_att=2.0)
        cls.result = PropensityScoreMatching(
            make_dag(), treatment="education", outcome="income"
        ).fit(cls.df)

    def test_returns_matching_result(self):
        assert isinstance(self.result, MatchingResult)

    def test_att_close_to_true_effect(self):
        assert abs(self.result.effect - 2.0) < 0.5

    def test_std_err_positive(self):
        assert self.result.std_err > 0

    def test_conf_int_is_ordered(self):
        lo, hi = self.result.conf_int
        assert lo < hi

    def test_conf_int_brackets_att(self):
        lo, hi = self.result.conf_int
        assert lo < self.result.effect < hi

    def test_pvalue_significant(self):
        assert self.result.pvalue < 0.05

    def test_adjustment_set_contains_ability(self):
        assert "ability" in self.result.adjustment_set

    def test_unadjusted_effect_differs_from_att(self):
        assert self.result.unadjusted_effect != self.result.effect

    def test_bootstrap_atts_array(self):
        boot = self.result.bootstrap_atts
        assert len(boot) > 0
        assert boot.ndim == 1

    def test_bootstrap_atts_returns_copy(self):
        boot = self.result.bootstrap_atts
        original_first = self.result.bootstrap_atts[0]
        boot[:] = 0
        assert self.result.bootstrap_atts[0] == original_first


class TestMatchingResultSummary:
    """Fit once, then test summary output."""

    @classmethod
    def setup_class(cls):
        cls.result = PropensityScoreMatching(
            make_dag(), treatment="education", outcome="income"
        ).fit(make_data())

    def test_summary_contains_att(self):
        summary = self.result.summary()
        assert "ATT" in summary
        assert "education" in summary
        assert "income" in summary

    def test_repr_is_summary(self):
        assert repr(self.result) == self.result.summary()


class TestMatchingNoConfounders:
    def test_fit_with_no_confounders(self):
        dag = DAG()
        dag.assume("education").causes("income")
        rng = np.random.default_rng(7)
        n = 200
        education = rng.choice([0, 1], size=n).astype(float)
        income    = 1.5 * education + rng.normal(size=n)
        df = pd.DataFrame({"education": education, "income": income})
        result = PropensityScoreMatching(dag, treatment="education", outcome="income").fit(df)
        assert isinstance(result, MatchingResult)
        assert result.adjustment_set == set()
