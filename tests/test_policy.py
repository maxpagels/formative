import numpy as np
import pandas as pd
import pytest

from formative.causal import DAG, RCT, PolicyNode, PolicyResult

RNG = np.random.default_rng(42)
N = 4_000


def make_policy_data():
    """
    Ground truth DGP with targetable heterogeneity:
      segment  ~ Uniform{0, 1, 2}     [modifies the effect]
      region   ~ {north, south}       [causes earnings, no effect modification]
      training ~ Bernoulli(0.5)       [randomised]
      earnings = 4*training*(segment == 2) + 0.5*segment + noise

    With cost=1, benefit=1 the per-unit net benefit of treating is −1 for
    segments 0/1 and +3 for segment 2, so the optimal policy is
    "treat iff segment = 2". Its value over the best constant policy
    (treat everyone, worth 1/3 per unit) is 1 − 1/3 = 2/3 per unit.
    """
    segment = RNG.integers(0, 3, size=N)
    region = RNG.choice(["north", "south"], size=N)
    training = RNG.integers(0, 2, size=N)
    earnings = 4.0 * training * (segment == 2) + 0.5 * segment + RNG.normal(size=N)
    return pd.DataFrame({"segment": segment, "region": region, "training": training, "earnings": earnings})


def make_null_data():
    """No effect anywhere: with cost=1 the optimal policy is to treat no one."""
    segment = RNG.integers(0, 3, size=N)
    region = RNG.choice(["north", "south"], size=N)
    training = RNG.integers(0, 2, size=N)
    earnings = 0.5 * segment + RNG.normal(size=N)
    return pd.DataFrame({"segment": segment, "region": region, "training": training, "earnings": earnings})


def make_dag():
    dag = DAG()
    dag.assume("training").causes("earnings")
    dag.assume("segment").causes("earnings")
    dag.assume("region").causes("earnings")
    return dag


class TestPolicyLearning:
    @classmethod
    def setup_class(cls):
        cls.df = make_policy_data()
        cls.result = RCT(make_dag(), treatment="training", outcome="earnings").fit(cls.df)
        cls.policy = cls.result.learn_policy(
            cls.df, modifiers=["segment", "region"], cost=1.0, benefit=1.0, max_depth=1
        )
        cls.policy2 = cls.result.learn_policy(
            cls.df, modifiers=["segment", "region"], cost=1.0, benefit=1.0, max_depth=2
        )

    def test_returns_policy_result(self):
        assert isinstance(self.policy, PolicyResult)

    def test_recovers_optimal_rule(self):
        assigned = self.policy.assign(self.df).to_numpy()
        assert (assigned == (self.df["segment"] == 2).to_numpy()).all()

    def test_tree_splits_on_segment(self):
        tree = self.policy.tree
        assert isinstance(tree, PolicyNode)
        assert tree.feature == "segment"
        assert tree.level == 2
        assert tree.if_true is True
        assert tree.if_false is False

    def test_value_near_truth(self):
        assert abs(self.policy.value - 2 / 3) < 0.2

    def test_value_uncertainty(self):
        lo, hi = self.policy.value_ci
        assert self.policy.value_se > 0
        assert lo < self.policy.value < hi

    def test_value_significantly_positive(self):
        assert self.policy.value_ci[0] > 0

    def test_coverage(self):
        assert abs(self.policy.coverage - 1 / 3) < 0.05

    def test_assign_is_aligned_boolean_series(self):
        new = self.df.head(10).copy()
        new.index = [f"unit_{i}" for i in range(10)]
        assigned = self.policy.assign(new)
        assert isinstance(assigned, pd.Series)
        assert assigned.dtype == bool
        assert list(assigned.index) == list(new.index)
        assert assigned.name == "treat"

    def test_rules_text(self):
        assert "treat if segment = 2" in self.policy.rules
        assert "otherwise don't treat" in self.policy.rules

    def test_depth_two_does_not_lose_value(self):
        assert self.policy2.value > 0
        assert abs(self.policy2.value - self.policy.value) < 0.2

    def test_modifiers_property(self):
        assert self.policy.modifiers == ["segment", "region"]

    def test_assumptions(self):
        names = [a.name for a in self.policy.assumptions]
        assert any("Random assignment" in n for n in names)
        assert any("pre-treatment" in n for n in names)
        assert any("Cost and benefit" in n for n in names)

    def test_summary(self):
        summary = self.policy.summary()
        assert "Learned Policy: training → earnings" in summary
        assert "treat if segment = 2" in summary
        assert "Value vs constant" in summary
        assert "Coverage" in summary
        assert "Assumptions" in summary

    def test_repr_is_summary(self):
        assert repr(self.policy) == self.policy.summary()

    def test_executive_summary(self):
        text = self.policy.executive_summary()
        assert "Learned Treatment Policy" in text
        assert "treat if segment = 2" in text
        assert "METHOD" in text
        assert "CAVEATS" in text

    def test_learn_policy_from_cate_result(self):
        result = RCT(make_dag(), treatment="training", outcome="earnings", effect_modifier="segment").fit(self.df)
        policy = result.learn_policy(self.df, modifiers=["segment"], cost=1.0, benefit=1.0, max_depth=1)
        assert isinstance(policy, PolicyResult)


class TestPolicyMultipleModifiers:
    """
    DGP where the optimal rule needs both features:
      earnings = 5*training*(segment == 2 AND region == north) + 0.5*segment + noise

    With cost=1, benefit=1 the net benefit of treating is +4 in the
    segment-2-north cell (1/6 of the sample) and −1 everywhere else. The best
    constant policy is treat-no-one (mean γ = 4/6 − 5/6 < 0), so the optimal
    depth-2 policy is worth (1/6)·4 = 2/3 per unit. A depth-1 tree cannot
    express the interaction: its best rule treats all of segment 2, worth
    only (1/3)·(4−1)/2 = 1/2 per unit.
    """

    @classmethod
    def setup_class(cls):
        rng = np.random.default_rng(11)
        segment = rng.integers(0, 3, size=N)
        region = rng.choice(["north", "south"], size=N)
        training = rng.integers(0, 2, size=N)
        target = (segment == 2) & (region == "north")
        earnings = 5.0 * training * target + 0.5 * segment + rng.normal(size=N)
        cls.df = pd.DataFrame({"segment": segment, "region": region, "training": training, "earnings": earnings})
        cls.target = target
        result = RCT(make_dag(), treatment="training", outcome="earnings").fit(cls.df)
        cls.policy1 = result.learn_policy(cls.df, modifiers=["segment", "region"], cost=1.0, benefit=1.0, max_depth=1)
        cls.policy2 = result.learn_policy(cls.df, modifiers=["segment", "region"], cost=1.0, benefit=1.0, max_depth=2)

    def test_depth_two_recovers_interaction_rule(self):
        assert (self.policy2.assign(self.df).to_numpy() == self.target).all()

    def test_rules_mention_both_features(self):
        assert "segment = 2" in self.policy2.rules
        assert "region" in self.policy2.rules

    def test_value_near_truth(self):
        assert abs(self.policy2.value - 2 / 3) < 0.2

    def test_coverage_is_target_cell(self):
        assert abs(self.policy2.coverage - 1 / 6) < 0.03

    def test_depth_one_cannot_express_interaction(self):
        assert (self.policy1.assign(self.df).to_numpy() == (self.df["segment"] == 2).to_numpy()).all()
        assert self.policy2.value > self.policy1.value


class TestPolicySimplification:
    def test_redundant_conditions_are_pruned(self):
        """
        With three tenure bands and signal only in one, an equal-value depth-2
        tree with a redundant root (e.g. "tenure ≠ >5y and tenure = <2y") can
        win the search on candidate order; simplification must collapse it.
        """
        rng = np.random.default_rng(7)
        n = 5_000
        tenure = rng.choice(["<2y", "2-5y", ">5y"], size=n)
        coaching = rng.integers(0, 2, size=n)
        retention = 6.0 * coaching * (tenure == "<2y") + 2.0 * (tenure == ">5y") + rng.normal(size=n)
        df = pd.DataFrame({"tenure": tenure, "coaching": coaching, "retention": retention})

        dag = DAG()
        dag.assume("coaching").causes("retention")
        dag.assume("tenure").causes("retention")

        result = RCT(dag, treatment="coaching", outcome="retention").fit(df)
        policy = result.learn_policy(df, modifiers=["tenure"], cost=2.5, benefit=1.0, max_depth=2)
        assert policy.rules == "treat if tenure = <2y\notherwise don't treat"


class TestPolicyNull:
    @classmethod
    def setup_class(cls):
        cls.df = make_null_data()
        cls.result = RCT(make_dag(), treatment="training", outcome="earnings").fit(cls.df)
        cls.policy = cls.result.learn_policy(
            cls.df, modifiers=["segment", "region"], cost=1.0, benefit=1.0, max_depth=2
        )

    def test_treats_no_one(self):
        assert self.policy.tree is False
        assert self.policy.coverage == 0.0
        assert self.policy.rules == "treat no one"

    def test_value_is_zero(self):
        assert self.policy.value == 0.0

    def test_assign_all_false(self):
        assert not self.policy.assign(self.df).any()


class TestPolicyValidation:
    @classmethod
    def setup_class(cls):
        cls.df = make_policy_data()
        cls.result = RCT(make_dag(), treatment="training", outcome="earnings").fit(cls.df)

    def learn(self, df=None, **overrides):
        kwargs = dict(modifiers=["segment"], cost=1.0, benefit=1.0, max_depth=1)
        kwargs.update(overrides)
        return self.result.learn_policy(self.df if df is None else df, **kwargs)

    def test_empty_modifiers_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            self.learn(modifiers=[])

    def test_duplicate_modifiers_raises(self):
        with pytest.raises(ValueError, match="duplicates"):
            self.learn(modifiers=["segment", "segment"])

    @pytest.mark.parametrize("depth", [0, 3])
    def test_bad_depth_raises(self, depth):
        with pytest.raises(ValueError, match="max_depth"):
            self.learn(max_depth=depth)

    def test_modifier_not_in_dag_raises(self):
        with pytest.raises(ValueError, match="not a node in the DAG"):
            self.learn(modifiers=["tenure"])

    def test_mediator_modifier_raises(self):
        dag = make_dag()
        dag.assume("training").causes("morale")
        dag.assume("morale").causes("earnings")
        df = self.df.assign(morale=RNG.integers(0, 2, size=N))
        result = RCT(dag, treatment="training", outcome="earnings").fit(df)
        with pytest.raises(ValueError, match="descendant of treatment"):
            result.learn_policy(df, modifiers=["morale"], cost=1.0, benefit=1.0)

    def test_modifier_not_outcome_ancestor_raises(self):
        dag = make_dag()
        dag.assume("hair_colour").causes("style")
        df = self.df.assign(hair_colour=RNG.integers(0, 2, size=N))
        result = RCT(dag, treatment="training", outcome="earnings").fit(df)
        with pytest.raises(ValueError, match="not an ancestor of outcome"):
            result.learn_policy(df, modifiers=["hair_colour"], cost=1.0, benefit=1.0)

    def test_modifier_column_missing_raises(self):
        dag = make_dag()
        dag.assume("tenure").causes("earnings")
        result = RCT(dag, treatment="training", outcome="earnings").fit(self.df)
        with pytest.raises(ValueError, match="not found in dataframe"):
            result.learn_policy(self.df, modifiers=["tenure"], cost=1.0, benefit=1.0)

    def test_non_binary_treatment_raises(self):
        df = self.df.assign(training=RNG.integers(0, 3, size=N))
        with pytest.raises(ValueError, match="binary 0/1 treatment"):
            self.learn(df=df)

    def test_missing_values_raise(self):
        df = self.df.copy()
        df.loc[df.index[0], "segment"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            self.learn(df=df)

    def test_single_level_modifier_raises(self):
        df = self.df.assign(segment=1)
        with pytest.raises(ValueError, match="at least 2 levels"):
            self.learn(df=df)

    def test_too_many_levels_raises(self):
        df = self.df.assign(segment=np.arange(N))
        with pytest.raises(ValueError, match="bin continuous"):
            self.learn(df=df)

    def test_too_few_units_per_arm_raises(self):
        df = self.df.head(12)
        with pytest.raises(ValueError, match="each treatment arm"):
            self.learn(df=df)
