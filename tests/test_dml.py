import numpy as np
import pandas as pd
import pytest

pytest.importorskip("doubleml")

from sklearn.linear_model import LinearRegression  # noqa: E402

from formative.causal import DAG, DoubleML, DoubleMLResult, OLSObservational  # noqa: E402
from formative.causal._exceptions import IdentificationError  # noqa: E402

N = 1_000
TRUE_EFFECT = 2.0


def make_dag():
    dag = DAG()
    dag.assume("ability").causes("education", "income")
    dag.assume("education").causes("income")
    return dag


def make_data(binary=False, selection=0.5):
    """
    Confounding through ability² — invisible to a linear control for ability.
    ``selection`` sets how strongly ability² drives treatment; at 1.0 the
    binary treatment has poor overlap (extreme ability → always treated).
    """
    rng = np.random.default_rng(42)
    ability = rng.normal(size=N)
    latent = selection * ability**2 + rng.normal(size=N)
    education = (latent > np.median(latent)).astype(float) if binary else latent
    income = TRUE_EFFECT * education + 2.0 * ability**2 + rng.normal(size=N)
    return pd.DataFrame({"ability": ability, "education": education, "income": income})


class TestDoubleMLValidation:
    def test_treatment_not_in_dag_raises(self):
        with pytest.raises(ValueError, match="not a node"):
            DoubleML(make_dag(), treatment="x", outcome="income")

    def test_same_treatment_and_outcome_raises(self):
        with pytest.raises(ValueError, match="different"):
            DoubleML(make_dag(), treatment="income", outcome="income")

    def test_too_few_folds_raises(self):
        with pytest.raises(ValueError, match="n_folds"):
            DoubleML(make_dag(), treatment="education", outcome="income", n_folds=1)

    def test_missing_confounder_raises(self):
        df = make_data().drop(columns="ability")
        with pytest.raises(IdentificationError):
            DoubleML(make_dag(), treatment="education", outcome="income").fit(df)

    def test_constant_treatment_raises(self):
        df = make_data().assign(education=1.0)
        with pytest.raises(ValueError, match="only one value"):
            DoubleML(make_dag(), treatment="education", outcome="income").fit(df)

    def test_no_confounders_raises(self):
        dag = DAG()
        dag.assume("education").causes("income")
        with pytest.raises(ValueError, match="no confounders"):
            DoubleML(dag, treatment="education", outcome="income").fit(make_data())


class TestDoubleMLContinuous:
    @classmethod
    def setup_class(cls):
        cls.df = make_data()
        cls.result = DoubleML(make_dag(), treatment="education", outcome="income").fit(cls.df)

    def test_returns_result(self):
        assert isinstance(self.result, DoubleMLResult)
        assert type(self.result.doubleml_model).__name__ == "DoubleMLPLR"

    def test_recovers_effect_where_ols_is_biased(self):
        ols = OLSObservational(make_dag(), treatment="education", outcome="income").fit(self.df)
        assert abs(self.result.effect - TRUE_EFFECT) < 0.2
        assert abs(ols.effect - TRUE_EFFECT) > 0.5

    def test_ci_covers_truth(self):
        lo, hi = self.result.conf_int
        assert lo < TRUE_EFFECT < hi
        assert self.result.std_err > 0
        assert self.result.pvalue < 0.05

    def test_deterministic(self):
        again = DoubleML(make_dag(), treatment="education", outcome="income").fit(self.df)
        assert again.effect == self.result.effect

    def test_summary(self):
        s = self.result.summary()
        assert "Double ML Causal Effect (PLR)" in s
        assert "ability" in s

    def test_executive_summary(self):
        s = self.result.executive_summary()
        assert "Double Machine Learning" in s
        assert "average slope" in s

    def test_refutations_pass(self):
        report = self.result.refute(self.df)
        assert [c.name for c in report.checks] == ["Random common cause", "Placebo treatment", "Nuisance model fit"]
        assert report.passed, report.summary()


class TestDoubleMLBinary:
    @classmethod
    def setup_class(cls):
        cls.df = make_data(binary=True)
        cls.result = DoubleML(make_dag(), treatment="education", outcome="income").fit(cls.df)

    def test_uses_irm(self):
        assert type(self.result.doubleml_model).__name__ == "DoubleMLIRM"

    def test_recovers_effect(self):
        lo, hi = self.result.conf_int
        assert lo < TRUE_EFFECT < hi

    def test_executive_summary(self):
        assert "treating everyone with education" in self.result.executive_summary()

    def test_refutations_pass(self):
        report = self.result.refute(self.df)
        assert report.checks[-1].name == "Overlap"
        assert report.passed, report.summary()


def test_poor_overlap_fails_check():
    df = make_data(binary=True, selection=1.0)
    report = DoubleML(make_dag(), treatment="education", outcome="income").fit(df).refute(df)
    overlap = next(c for c in report.checks if c.name == "Overlap")
    assert not overlap.passed, overlap.detail


def test_near_deterministic_treatment_fails_nuisance_check():
    rng = np.random.default_rng(0)
    ability = rng.normal(size=N)
    education = ability + 0.02 * rng.normal(size=N)
    income = TRUE_EFFECT * education + ability + rng.normal(size=N)
    df = pd.DataFrame({"ability": ability, "education": education, "income": income})
    report = DoubleML(make_dag(), treatment="education", outcome="income").fit(df).refute(df)
    fit = next(c for c in report.checks if c.name == "Nuisance model fit")
    assert not fit.passed
    assert "almost fully determine" in fit.detail


def test_custom_learners():
    result = DoubleML(
        make_dag(),
        treatment="education",
        outcome="income",
        ml_outcome=LinearRegression(),
        ml_treatment=LinearRegression(),
    ).fit(make_data())
    # Linear nuisance models can't see ability², so the estimate stays biased like OLS.
    assert abs(result.effect - TRUE_EFFECT) > 0.5


def test_text_confounder_is_one_hot_encoded():
    rng = np.random.default_rng(0)
    region = rng.choice(["north", "south"], N)
    north = (region == "north").astype(float)
    spend = north + rng.normal(size=N)
    sales = TRUE_EFFECT * spend + north + rng.normal(size=N)
    df = pd.DataFrame({"region": region, "spend": spend, "sales": sales})
    dag = DAG()
    dag.assume("region").causes("spend", "sales")
    dag.assume("spend").causes("sales")
    lo, hi = DoubleML(dag, treatment="spend", outcome="sales").fit(df).conf_int
    assert lo < TRUE_EFFECT < hi


def test_overfit_outcome_model_fails_nuisance_check():
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(0)
    ability = rng.normal(size=N)
    education = ability + rng.normal(size=N)
    income = rng.normal(size=N)  # pure noise: an unpruned tree can only memorise it
    df = pd.DataFrame({"ability": ability, "education": education, "income": income})
    result = DoubleML(
        make_dag(), treatment="education", outcome="income", ml_outcome=DecisionTreeRegressor(random_state=0)
    ).fit(df)
    fit = next(c for c in result.refute(df).checks if c.name == "Nuisance model fit")
    assert not fit.passed
    assert "worse than the mean" in fit.detail
