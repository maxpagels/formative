import numpy as np
import pandas as pd
import pytest

from formative import DAG, OLSObservational
from formative._exceptions import IdentificationError


RNG = np.random.default_rng(42)
N = 1_000


def make_data(true_effect=2.0):
    """
    Ground truth DGP:
      ability   ~ N(0, 1)          [confounder — included or excluded from df to test]
      education = 0.5*ability + noise
      income    = true_effect*education + 0.8*ability + noise
    """
    ability = RNG.normal(size=N)
    education = 0.5 * ability + RNG.normal(size=N)
    income = true_effect * education + 0.8 * ability + RNG.normal(size=N)
    return pd.DataFrame({"ability": ability, "education": education, "income": income})


class TestDAG:
    def test_basic_edges(self):
        dag = DAG()
        dag.causes("A", "B").causes("B", "C")
        assert dag.parents("B") == {"A"}
        assert dag.children("B") == {"C"}
        assert dag.ancestors("C") == {"A", "B"}
        assert dag.descendants("A") == {"B", "C"}

    def test_cycle_detection(self):
        dag = DAG()
        dag.causes("A", "B").causes("B", "C")
        with pytest.raises(Exception, match="cycle"):
            dag.causes("C", "A")


class TestOLSObservational:
    def test_no_confounders_runs(self):
        """When there are no confounders in the DAG, OLS is just bivariate."""
        dag = DAG()
        dag.causes("education", "income")

        df = make_data(true_effect=2.0)
        result = OLSObservational(dag, treatment="education", outcome="income").fit(df)

        # Without controlling for ability, estimate will be biased — but it should run
        assert result.effect is not None
        assert result.adjustment_set == set()

    def test_observed_confounder_is_controlled(self):
        """With ability in the dataframe, it is added to the adjustment set automatically."""
        dag = DAG()
        dag.causes("ability", "education")
        dag.causes("ability", "income")
        dag.causes("education", "income")

        df = make_data(true_effect=2.0)
        result = OLSObservational(dag, treatment="education", outcome="income").fit(df)

        assert "ability" in result.adjustment_set
        assert abs(result.effect - 2.0) < 0.15  # should recover the true effect

    def test_unobserved_confounder_raises(self):
        """With ability absent from the dataframe, OLS should refuse to run."""
        dag = DAG()
        dag.causes("ability", "education")
        dag.causes("ability", "income")
        dag.causes("education", "income")

        df = make_data(true_effect=2.0).drop(columns=["ability"])
        with pytest.raises(IdentificationError, match="Unobserved confounders"):
            OLSObservational(dag, treatment="education", outcome="income").fit(df)

    def test_mediator_not_in_adjustment_set(self):
        """
        Mediators are descendants of treatment and must not be controlled for —
        doing so would block the causal path and underestimate the effect.
        """
        dag = DAG()
        dag.causes("education", "skills")   # mediator
        dag.causes("skills", "income")
        dag.causes("education", "income")   # direct path too

        df = make_data(true_effect=2.0)
        result = OLSObservational(dag, treatment="education", outcome="income").fit(df)

        assert "skills" not in result.adjustment_set

    def test_unknown_treatment_raises(self):
        dag = DAG()
        dag.causes("A", "B")
        with pytest.raises(ValueError, match="Treatment"):
            OLSObservational(dag, treatment="X", outcome="B")

    def test_missing_treatment_column_raises(self):
        dag = DAG()
        dag.causes("ability", "education")
        dag.causes("ability", "income")
        dag.causes("education", "income")

        df = make_data().drop(columns=["education"])
        with pytest.raises(ValueError, match="Treatment column"):
            OLSObservational(dag, treatment="education", outcome="income").fit(df)

    def test_result_has_expected_attributes(self):
        dag = DAG()
        dag.causes("ability", "education")
        dag.causes("ability", "income")
        dag.causes("education", "income")

        df = make_data()
        result = OLSObservational(dag, treatment="education", outcome="income").fit(df)

        lo, hi = result.conf_int
        assert lo < result.effect < hi
        assert result.std_err > 0
        assert 0 <= result.pvalue <= 1
        assert result.statsmodels_result is not None
        assert result.statsmodels_unadjusted_result is not None

    def test_unadjusted_estimate_is_biased(self):
        """When ability is a confounder, unadjusted OLS overestimates the true effect."""
        dag = DAG()
        dag.causes("ability", "education")
        dag.causes("ability", "income")
        dag.causes("education", "income")

        df = make_data(true_effect=2.0)
        result = OLSObservational(dag, treatment="education", outcome="income").fit(df)

        assert abs(result.effect - 2.0) < 0.15           # adjusted: close to truth
        assert abs(result.unadjusted_effect - 2.0) > 0.1  # unadjusted: meaningfully off
        assert result.unadjusted_effect > result.effect    # ability inflates the estimate

    def test_no_confounders_adjusted_equals_unadjusted(self):
        """With no confounders in the DAG, both estimates should be identical."""
        dag = DAG()
        dag.causes("education", "income")

        df = make_data()
        result = OLSObservational(dag, treatment="education", outcome="income").fit(df)

        assert result.effect == result.unadjusted_effect
