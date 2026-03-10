from __future__ import annotations

from formative.causal.decision import DecisionReport, _decide

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TREATMENT = "training"
OUTCOME = "earnings"

# A clearly profitable decision: effect=3, benefit=15, cost=8 → net=37
POSITIVE = _decide(
    effect=3.0,
    std_err=0.5,
    conf_int=(2.02, 3.98),
    treatment=TREATMENT,
    outcome=OUTCOME,
    cost=8.0,
    benefit=15.0,
)

# A clearly unprofitable decision: effect=0.1, benefit=5, cost=10 → net=-9.5
NEGATIVE = _decide(
    effect=0.1,
    std_err=0.2,
    conf_int=(-0.29, 0.49),
    treatment=TREATMENT,
    outcome=OUTCOME,
    cost=10.0,
    benefit=5.0,
)

# A marginal decision: CI straddles zero → not robust
MARGINAL = _decide(
    effect=1.0,
    std_err=1.0,
    conf_int=(-0.96, 2.96),
    treatment=TREATMENT,
    outcome=OUTCOME,
    cost=10.0,
    benefit=8.0,
)


# ---------------------------------------------------------------------------
# _decide — net benefit arithmetic
# ---------------------------------------------------------------------------


class TestDecideNetBenefit:
    def test_net_benefit_positive(self):
        assert abs(POSITIVE.net_benefit - (3.0 * 15.0 - 8.0)) < 1e-9

    def test_net_benefit_negative(self):
        assert abs(NEGATIVE.net_benefit - (0.1 * 5.0 - 10.0)) < 1e-9

    def test_ci_propagated(self):
        lo, hi = POSITIVE.net_benefit_ci
        assert abs(lo - (2.02 * 15.0 - 8.0)) < 1e-9
        assert abs(hi - (3.98 * 15.0 - 8.0)) < 1e-9


# ---------------------------------------------------------------------------
# _decide — optimal decision
# ---------------------------------------------------------------------------


class TestDecideOptimal:
    def test_treat_when_positive(self):
        assert POSITIVE.optimal == "treat"

    def test_dont_treat_when_negative(self):
        assert NEGATIVE.optimal == "don't treat"

    def test_zero_net_benefit_is_dont_treat(self):
        # effect * benefit == cost exactly → net_benefit == 0 → "don't treat"
        report = _decide(2.0, 0.1, (1.8, 2.2), TREATMENT, OUTCOME, cost=10.0, benefit=5.0)
        assert report.optimal == "don't treat"


# ---------------------------------------------------------------------------
# _decide — robustness
# ---------------------------------------------------------------------------


class TestDecideRobust:
    def test_robust_when_ci_wholly_positive(self):
        assert POSITIVE.robust is True

    def test_robust_when_ci_wholly_negative(self):
        assert NEGATIVE.robust is True

    def test_not_robust_when_ci_straddles_zero(self):
        assert MARGINAL.robust is False


# ---------------------------------------------------------------------------
# _decide — p_beneficial
# ---------------------------------------------------------------------------


class TestDecidePBeneficial:
    def test_high_confidence_for_clear_positive(self):
        assert POSITIVE.p_beneficial > 0.99

    def test_low_confidence_for_clear_negative(self):
        assert NEGATIVE.p_beneficial < 0.01

    def test_near_half_for_marginal(self):
        # net_benefit = 1*8 - 10 = -2; se_net = 1*8 = 8 → p ≈ norm.cdf(-2/8) ≈ 0.40
        assert 0.3 < MARGINAL.p_beneficial < 0.5

    def test_zero_se_positive_effect(self):
        report = _decide(2.0, 0.0, (2.0, 2.0), TREATMENT, OUTCOME, cost=1.0, benefit=5.0)
        assert report.p_beneficial == 1.0

    def test_zero_se_negative_effect(self):
        report = _decide(0.1, 0.0, (0.1, 0.1), TREATMENT, OUTCOME, cost=10.0, benefit=5.0)
        assert report.p_beneficial == 0.0


# ---------------------------------------------------------------------------
# _decide — metadata fields
# ---------------------------------------------------------------------------


class TestDecideFields:
    def test_treatment_and_outcome_stored(self):
        assert POSITIVE.treatment == TREATMENT
        assert POSITIVE.outcome == OUTCOME

    def test_cost_and_benefit_stored(self):
        assert POSITIVE.cost == 8.0
        assert POSITIVE.benefit == 15.0

    def test_returns_decision_report(self):
        assert isinstance(POSITIVE, DecisionReport)


# ---------------------------------------------------------------------------
# value_of_information
# ---------------------------------------------------------------------------


class TestValueOfInformation:
    def test_already_confident_returns_no_data_needed(self):
        msg = POSITIVE.value_of_information(target_confidence=0.95)
        assert "No additional data is needed" in msg

    def test_below_target_reports_shrinkage_and_multiplier(self):
        msg = MARGINAL.value_of_information(target_confidence=0.95)
        assert "standard error" in msg
        assert "sample size" in msg

    def test_zero_se_returns_error_message(self):
        report = _decide(0.0, 0.0, (0.0, 0.0), TREATMENT, OUTCOME, cost=1.0, benefit=5.0)
        msg = report.value_of_information()
        assert "standard error is zero" in msg

    def test_custom_target_confidence(self):
        msg = POSITIVE.value_of_information(target_confidence=0.9999)
        # Even a very profitable decision may need more data at 99.99% confidence
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# summary / __repr__
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_contains_treatment_and_outcome(self):
        s = POSITIVE.summary()
        assert TREATMENT in s
        assert OUTCOME in s

    def test_summary_contains_optimal_decision(self):
        assert "treat" in POSITIVE.summary()
        assert "don't treat" in NEGATIVE.summary()

    def test_summary_contains_net_benefit(self):
        assert "37" in POSITIVE.summary()

    def test_robust_label_in_summary(self):
        assert "stable" in POSITIVE.summary()
        assert "flips" in MARGINAL.summary()

    def test_repr_equals_summary(self):
        assert repr(POSITIVE) == POSITIVE.summary()
