import numpy as np
import pandas as pd
import pytest

from formative.causal import (
    DAG,
    RCT,
    DecisionReport,
    GroupEffect,
    OLSCATEResult,
    OLSObservational,
    RCTCATEResult,
)

RNG = np.random.default_rng(42)
N = 2_000


def make_heterogeneous_data():
    """
    Ground truth DGP with effect modification:
      segment   ~ Uniform{0, 1, 2}   [effect modifier — causes income, not education]
      ability   ~ N(0, 1)            [confounder]
      education = ability + noise
      income    = (1 + segment)*education + 1.5*ability + 0.5*segment + noise
    True group effects: 1.0, 2.0, 3.0.
    """
    segment = RNG.integers(0, 3, size=N)
    ability = RNG.normal(size=N)
    education = ability + RNG.normal(size=N)
    income = (1.0 + segment) * education + 1.5 * ability + 0.5 * segment + RNG.normal(size=N)
    return pd.DataFrame({"segment": segment, "ability": ability, "education": education, "income": income})


def make_homogeneous_data():
    """Same DGP but with a constant effect of 2.0 across all segments."""
    segment = RNG.integers(0, 3, size=N)
    ability = RNG.normal(size=N)
    education = ability + RNG.normal(size=N)
    income = 2.0 * education + 1.5 * ability + 0.5 * segment + RNG.normal(size=N)
    return pd.DataFrame({"segment": segment, "ability": ability, "education": education, "income": income})


def make_dag():
    dag = DAG()
    dag.assume("ability").causes("education", "income")
    dag.assume("segment").causes("income")
    dag.assume("education").causes("income")
    return dag


class TestOLSCATEEstimation:
    @classmethod
    def setup_class(cls):
        cls.df = make_heterogeneous_data()
        cls.result = OLSObservational(
            make_dag(), treatment="education", outcome="income", effect_modifier="segment"
        ).fit(cls.df)

    def test_returns_cate_result(self):
        assert isinstance(self.result, OLSCATEResult)

    def test_recovers_group_effects(self):
        effects = {g.level: g.effect for g in self.result.effect_by_group}
        assert abs(effects[0] - 1.0) < 0.15
        assert abs(effects[1] - 2.0) < 0.15
        assert abs(effects[2] - 3.0) < 0.15

    def test_group_effects_are_group_effect_instances(self):
        for g in self.result.effect_by_group:
            assert isinstance(g, GroupEffect)
            assert g.std_err > 0
            assert g.conf_int[0] < g.effect < g.conf_int[1]

    def test_group_ns_sum_to_total(self):
        assert sum(g.n for g in self.result.effect_by_group) == N

    def test_weighted_effect_is_share_weighted_average(self):
        weighted = sum(g.effect * g.n for g in self.result.effect_by_group) / N
        assert abs(self.result.effect - weighted) < 1e-10

    def test_heterogeneity_detected(self):
        assert self.result.homogeneity_pvalue < 0.001
        assert self.result.homogeneity_fstat > 10

    def test_effect_modifier_name(self):
        assert self.result.effect_modifier == "segment"

    def test_confounder_still_controlled(self):
        assert "ability" in self.result.adjustment_set

    def test_cate_assumptions_appended(self):
        names = [a.name for a in self.result.assumptions]
        assert any("effect modification" in n for n in names)
        assert any("pre-treatment" in n for n in names)

    def test_summary_contains_group_block(self):
        summary = self.result.summary()
        assert "Effect by segment" in summary
        assert "homogeneity" in summary

    def test_repr_is_summary(self):
        assert repr(self.result) == self.result.summary()

    def test_executive_summary_contains_heterogeneity(self):
        assert "HETEROGENEITY" in self.result.executive_summary()

    def test_decide_by_group(self):
        decisions = self.result.decide_by_group(cost=2.5, benefit=1.0)
        assert set(decisions) == {0, 1, 2}
        assert all(isinstance(d, DecisionReport) for d in decisions.values())
        # True effects 1, 2, 3 with cost 2.5: only segment 2 is worth treating.
        assert decisions[0].optimal == "don't treat"
        assert decisions[2].optimal == "treat"

    def test_refutations_pass_on_well_specified_data(self):
        report = self.result.refute(self.df)
        names = [c.name for c in report.checks]
        assert names == ["Random common cause", "Placebo modifier", "Random modifier"]
        assert report.passed


class TestOLSCATEHomogeneous:
    def test_no_false_heterogeneity(self):
        df = make_homogeneous_data()
        result = OLSObservational(make_dag(), treatment="education", outcome="income", effect_modifier="segment").fit(
            df
        )
        assert result.homogeneity_pvalue > 0.05
        assert abs(result.effect - 2.0) < 0.15


class TestRCTCATEEstimation:
    @classmethod
    def setup_class(cls):
        segment = RNG.integers(0, 3, size=N)
        treatment = RNG.integers(0, 2, size=N).astype(float)
        outcome = (1.0 + 2.0 * (segment == 2)) * treatment + 0.3 * segment + RNG.normal(size=N)
        cls.df = pd.DataFrame({"segment": segment, "treatment": treatment, "outcome": outcome})

        dag = DAG()
        dag.assume("treatment").causes("outcome")
        dag.assume("segment").causes("outcome")
        cls.result = RCT(dag, treatment="treatment", outcome="outcome", effect_modifier="segment").fit(cls.df)

    def test_returns_cate_result(self):
        assert isinstance(self.result, RCTCATEResult)

    def test_recovers_group_effects(self):
        effects = {g.level: g.effect for g in self.result.effect_by_group}
        assert abs(effects[0] - 1.0) < 0.2
        assert abs(effects[1] - 1.0) < 0.2
        assert abs(effects[2] - 3.0) < 0.2

    def test_heterogeneity_detected(self):
        assert self.result.homogeneity_pvalue < 0.001

    def test_summary_contains_group_block(self):
        assert "Effect by segment" in self.result.summary()

    def test_executive_summary_contains_heterogeneity(self):
        assert "HETEROGENEITY" in self.result.executive_summary()

    def test_refutations_pass_on_well_specified_data(self):
        assert self.result.refute(self.df).passed


class TestCATEDagValidation:
    def test_modifier_not_in_dag_raises(self):
        with pytest.raises(ValueError, match="not a node in the DAG"):
            OLSObservational(make_dag(), treatment="education", outcome="income", effect_modifier="tenure")

    def test_modifier_equal_to_treatment_raises(self):
        with pytest.raises(ValueError, match="different from treatment and outcome"):
            OLSObservational(make_dag(), treatment="education", outcome="income", effect_modifier="education")

    def test_mediator_as_modifier_raises(self):
        """Segmenting by a post-treatment variable must be rejected."""
        dag = make_dag()
        dag.assume("education").causes("engagement")
        dag.assume("engagement").causes("income")
        with pytest.raises(ValueError, match="descendant of treatment"):
            OLSObservational(dag, treatment="education", outcome="income", effect_modifier="engagement")

    def test_modifier_not_causing_outcome_raises(self):
        dag = make_dag()
        dag.assume("region").causes("holiday_pay")  # in the DAG, but no path to income
        with pytest.raises(ValueError, match="not an ancestor of outcome"):
            OLSObservational(dag, treatment="education", outcome="income", effect_modifier="region")

    def test_rct_validates_modifier_too(self):
        dag = DAG()
        dag.assume("treatment").causes("outcome")
        with pytest.raises(ValueError, match="not a node in the DAG"):
            RCT(dag, treatment="treatment", outcome="outcome", effect_modifier="segment")


class TestCATEDataValidation:
    def test_missing_modifier_column_raises(self):
        df = make_heterogeneous_data().drop(columns=["segment"])
        estimator = OLSObservational(make_dag(), treatment="education", outcome="income", effect_modifier="segment")
        with pytest.raises(ValueError, match="not found in dataframe"):
            estimator.fit(df)

    def test_single_level_modifier_raises(self):
        df = make_heterogeneous_data()
        df["segment"] = 1
        estimator = OLSObservational(make_dag(), treatment="education", outcome="income", effect_modifier="segment")
        with pytest.raises(ValueError, match="at least 2 levels"):
            estimator.fit(df)

    def test_no_treatment_variation_within_level_raises(self):
        df = make_heterogeneous_data()
        df.loc[df["segment"] == 0, "education"] = 0.0
        estimator = OLSObservational(make_dag(), treatment="education", outcome="income", effect_modifier="segment")
        with pytest.raises(ValueError, match="does not vary within"):
            estimator.fit(df)

    def test_string_modifier_levels_work(self):
        df = make_heterogeneous_data()
        df["segment"] = df["segment"].map({0: "low", 1: "mid", 2: "high"})
        result = OLSObservational(make_dag(), treatment="education", outcome="income", effect_modifier="segment").fit(
            df
        )
        effects = {g.level: g.effect for g in result.effect_by_group}
        assert abs(effects["low"] - 1.0) < 0.15
        assert abs(effects["mid"] - 2.0) < 0.15
        assert abs(effects["high"] - 3.0) < 0.15
