import numpy as np
import pandas as pd
import pytest

from formative import DAG
from formative.estimators.iv import IV2SLS, IVResult


RNG = np.random.default_rng(42)
N = 5_000  # larger sample — IV needs more data for precision


def make_dag():
    dag = DAG()
    dag.assume("proximity").causes("education")
    dag.assume("ability").causes("education", "income")
    dag.assume("education").causes("income")
    return dag


def make_iv_data(true_effect=2.0, include_ability=False):
    """
    Ground truth DGP:
      proximity  ~ N(0,1)                         [instrument, exogenous]
      ability    ~ N(0,1)                         [unobserved confounder]
      education  = 0.5*proximity + 0.5*ability + noise
      income     = true_effect*education + 0.8*ability + noise
    """
    proximity = RNG.normal(size=N)
    ability = RNG.normal(size=N)
    education = 0.5 * proximity + 0.5 * ability + RNG.normal(size=N)
    income = true_effect * education + 0.8 * ability + RNG.normal(size=N)
    df = pd.DataFrame({"proximity": proximity, "education": education, "income": income})
    if include_ability:
        df["ability"] = ability
    return df


class TestIV2SLSValidation:
    def test_instrument_not_in_dag_raises(self):
        dag = make_dag()
        with pytest.raises(ValueError, match="Instrument"):
            IV2SLS(dag, treatment="education", outcome="income", instrument="unknown")

    def test_treatment_not_in_dag_raises(self):
        dag = make_dag()
        with pytest.raises(ValueError, match="Treatment"):
            IV2SLS(dag, treatment="unknown", outcome="income", instrument="proximity")

    def test_instrument_not_causing_treatment_raises(self):
        dag = DAG()
        dag.assume("education").causes("income")
        dag.assume("proximity").causes("income")  # causes outcome, not treatment
        with pytest.raises(ValueError, match="does not cause treatment"):
            IV2SLS(dag, treatment="education", outcome="income", instrument="proximity")

    def test_exclusion_restriction_violation_raises(self):
        dag = DAG()
        dag.assume("proximity").causes("education", "income")  # direct path to outcome
        dag.assume("education").causes("income")
        with pytest.raises(ValueError, match="[Ee]xclusion restriction"):
            IV2SLS(dag, treatment="education", outcome="income", instrument="proximity")

    def test_missing_treatment_column_raises(self):
        dag = make_dag()
        df = make_iv_data().drop(columns=["education"])
        with pytest.raises(ValueError, match="Treatment column"):
            IV2SLS(dag, treatment="education", outcome="income", instrument="proximity").fit(df)

    def test_missing_instrument_column_raises(self):
        dag = make_dag()
        df = make_iv_data().drop(columns=["proximity"])
        with pytest.raises(ValueError, match="Instrument column"):
            IV2SLS(dag, treatment="education", outcome="income", instrument="proximity").fit(df)

    def test_missing_outcome_column_raises(self):
        dag = make_dag()
        df = make_iv_data().drop(columns=["income"])
        with pytest.raises(ValueError, match="Outcome column"):
            IV2SLS(dag, treatment="education", outcome="income", instrument="proximity").fit(df)


class TestIV2SLSEstimation:
    def test_iv_recovers_true_effect(self):
        """IV should recover the true effect even with an unobserved confounder."""
        dag = make_dag()
        df = make_iv_data(true_effect=2.0)  # ability unobserved
        result = IV2SLS(
            dag, treatment="education", outcome="income", instrument="proximity"
        ).fit(df)
        assert abs(result.effect - 2.0) < 0.2

    def test_unobserved_confounder_does_not_raise(self):
        """IV handles unobserved confounders — no IdentificationError expected."""
        dag = make_dag()
        df = make_iv_data()  # ability absent from df
        result = IV2SLS(
            dag, treatment="education", outcome="income", instrument="proximity"
        ).fit(df)
        assert result.adjustment_set == set()

    def test_observed_confounder_included_as_control(self):
        """If ability is observed it should appear in the adjustment set."""
        dag = make_dag()
        df = make_iv_data(include_ability=True)
        result = IV2SLS(
            dag, treatment="education", outcome="income", instrument="proximity"
        ).fit(df)
        assert "ability" in result.adjustment_set

    def test_instrument_not_in_adjustment_set(self):
        """The instrument itself must never appear in the adjustment set."""
        dag = make_dag()
        df = make_iv_data()
        result = IV2SLS(
            dag, treatment="education", outcome="income", instrument="proximity"
        ).fit(df)
        assert "proximity" not in result.adjustment_set

    def test_result_has_expected_attributes(self):
        dag = make_dag()
        df = make_iv_data()
        result = IV2SLS(
            dag, treatment="education", outcome="income", instrument="proximity"
        ).fit(df)
        lo, hi = result.conf_int
        assert lo < result.effect < hi
        assert result.std_err > 0
        assert 0 <= result.pvalue <= 1
        assert result.statsmodels_result is not None
        assert result.statsmodels_unadjusted_result is not None

    def test_unadjusted_estimate_is_biased(self):
        """With an unobserved confounder, naive OLS overestimates the true effect."""
        dag = make_dag()
        df = make_iv_data(true_effect=2.0)
        result = IV2SLS(
            dag, treatment="education", outcome="income", instrument="proximity"
        ).fit(df)
        assert abs(result.effect - 2.0) < 0.2
        assert abs(result.unadjusted_effect - 2.0) > 0.1
        assert result.unadjusted_effect > result.effect
