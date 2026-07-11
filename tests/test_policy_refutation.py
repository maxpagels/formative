import numpy as np
import pandas as pd

from formative.causal import DAG, RCT, PolicyRefutationReport, RefutationCheck

RNG = np.random.default_rng(42)
N = 4_000


def make_policy_data():
    """Same DGP as test_policy: treat iff segment = 2 is optimal at cost=1, benefit=1."""
    segment = RNG.integers(0, 3, size=N)
    region = RNG.choice(["north", "south"], size=N)
    training = RNG.integers(0, 2, size=N)
    earnings = 4.0 * training * (segment == 2) + 0.5 * segment + RNG.normal(size=N)
    return pd.DataFrame({"segment": segment, "region": region, "training": training, "earnings": earnings})


def make_dag():
    dag = DAG()
    dag.assume("training").causes("earnings")
    dag.assume("segment").causes("earnings")
    dag.assume("region").causes("earnings")
    return dag


class TestPolicyRefutation:
    @classmethod
    def setup_class(cls):
        cls.df = make_policy_data()
        result = RCT(make_dag(), treatment="training", outcome="earnings").fit(cls.df)
        cls.policy = result.learn_policy(cls.df, modifiers=["segment", "region"], cost=1.0, benefit=1.0, max_depth=2)
        cls.report = cls.policy.refute(cls.df)

    def test_returns_policy_refutation_report(self):
        assert isinstance(self.report, PolicyRefutationReport)

    def test_runs_both_checks(self):
        names = [c.name for c in self.report.checks]
        assert names == ["Placebo modifiers", "Random modifier"]
        for check in self.report.checks:
            assert isinstance(check, RefutationCheck)

    def test_well_specified_policy_passes(self):
        assert self.report.passed
        assert self.report.failed_checks == []

    def test_placebo_policy_has_no_value(self):
        placebo = next(c for c in self.report.checks if c.name == "Placebo modifiers")
        assert placebo.passed
        assert "Not significantly positive" in placebo.detail

    def test_random_modifier_adds_no_value(self):
        random_mod = next(c for c in self.report.checks if c.name == "Random modifier")
        assert random_mod.passed
        assert "1 SE" in random_mod.detail

    def test_summary_format(self):
        summary = self.report.summary()
        assert "Policy Refutation Report: training → earnings" in summary
        assert "[PASS]" in summary
        assert "All checks passed." in summary
