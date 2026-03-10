import pytest

from formative.game import expected_value, hurwicz, laplace, maximax, maximin, minimax_regret

OUTCOMES = {
    "stocks": {"recession": -20, "stagnation": 5, "growth": 30},
    "bonds": {"recession": 5, "stagnation": 5, "growth": 7},
    "cash": {"recession": 2, "stagnation": 2, "growth": 2},
}


class TestMaximin:
    def test_choice(self):
        assert maximin(OUTCOMES).solve().choice == "bonds"

    def test_guaranteed(self):
        assert maximin(OUTCOMES).solve().guaranteed == 5

    def test_worst_cases(self):
        r = maximin(OUTCOMES).solve()
        assert r.worst_cases["stocks"] == -20
        assert r.worst_cases["bonds"] == 5
        assert r.worst_cases["cash"] == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            maximin({})

    def test_single_choice(self):
        r = maximin({"only": {"good": 10, "bad": -1}}).solve()
        assert r.choice == "only"
        assert r.guaranteed == -1

    def test_repr_marks_chosen(self):
        assert "← chosen" in repr(maximin(OUTCOMES).solve())


class TestMaximax:
    def test_choice(self):
        assert maximax(OUTCOMES).solve().choice == "stocks"

    def test_best_case(self):
        assert maximax(OUTCOMES).solve().best_case == 30

    def test_best_cases(self):
        r = maximax(OUTCOMES).solve()
        assert r.best_cases["stocks"] == 30
        assert r.best_cases["bonds"] == 7
        assert r.best_cases["cash"] == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            maximax({})

    def test_repr_marks_chosen(self):
        assert "← chosen" in repr(maximax(OUTCOMES).solve())


class TestMinimaxRegret:
    def test_choice(self):
        # In recession: bonds regret=0, stocks regret=25, cash regret=3
        # In stagnation: bonds regret=0, stocks regret=0, cash regret=3
        # In growth: stocks regret=0, bonds regret=23, cash regret=28
        # max regrets: stocks=25, bonds=23, cash=28 → bonds chosen
        assert minimax_regret(OUTCOMES).solve().choice == "bonds"

    def test_max_regret_value(self):
        r = minimax_regret(OUTCOMES).solve()
        assert r.max_regrets["stocks"] == 25  # recession: 5 - (-20)
        assert r.max_regrets["bonds"] == 23  # growth: 30 - 7
        assert r.max_regrets["cash"] == 28  # growth: 30 - 2

    def test_regret_table(self):
        r = minimax_regret(OUTCOMES).solve()
        assert r.regret_table["stocks"]["recession"] == 25  # 5 - (-20)
        assert r.regret_table["bonds"]["recession"] == 0  # best in recession
        assert r.regret_table["stocks"]["growth"] == 0  # best in growth

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            minimax_regret({})

    def test_repr_marks_chosen(self):
        assert "← chosen" in repr(minimax_regret(OUTCOMES).solve())

    def test_single_choice_zero_regret(self):
        r = minimax_regret({"only": {"good": 10, "bad": -1}}).solve()
        assert r.choice == "only"
        assert r.max_regret == 0


class TestHurwicz:
    def test_alpha_zero_matches_maximin(self):
        h = hurwicz(OUTCOMES, alpha=0).solve()
        m = maximin(OUTCOMES).solve()
        assert h.choice == m.choice
        assert h.scores == pytest.approx(m.worst_cases)

    def test_alpha_one_matches_maximax(self):
        h = hurwicz(OUTCOMES, alpha=1).solve()
        m = maximax(OUTCOMES).solve()
        assert h.choice == m.choice
        assert h.scores == pytest.approx(m.best_cases)

    def test_alpha_half(self):
        # stocks: 0.5*30 + 0.5*(-20) = 5
        # bonds:  0.5*7  + 0.5*5    = 6
        # cash:   0.5*2  + 0.5*2    = 2
        # → bonds chosen at alpha=0.5
        r = hurwicz(OUTCOMES, alpha=0.5).solve()
        assert r.choice == "bonds"

    def test_scores(self):
        r = hurwicz(OUTCOMES, alpha=0.5).solve()
        assert r.scores["stocks"] == pytest.approx(5.0)
        assert r.scores["bonds"] == pytest.approx(6.0)
        assert r.scores["cash"] == pytest.approx(2.0)

    def test_alpha_stored_on_result(self):
        r = hurwicz(OUTCOMES, alpha=0.3).solve()
        assert r.alpha == pytest.approx(0.3)

    def test_score_matches_chosen_scores_entry(self):
        r = hurwicz(OUTCOMES, alpha=0.5).solve()
        assert r.score == pytest.approx(r.scores[r.choice])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            hurwicz({}, alpha=0.5)

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError):
            hurwicz(OUTCOMES, alpha=1.5)

    def test_repr_marks_chosen(self):
        assert "← chosen" in repr(hurwicz(OUTCOMES, alpha=0.5).solve())


class TestLaplace:
    def test_choice(self):
        # stocks: (-20+5+30)/3 = 5, bonds: (5+5+7)/3 = 5.67, cash: 2 → bonds
        # wait: stocks = 15/3 = 5, bonds = 17/3 ≈ 5.67 → bonds
        # Actually stocks: (-20+5+30)/3 = 15/3 = 5
        # bonds: (5+5+7)/3 = 17/3 ≈ 5.67 → bonds chosen
        assert laplace(OUTCOMES).solve().choice == "bonds"

    def test_averages(self):
        r = laplace(OUTCOMES).solve()
        assert r.averages["stocks"] == pytest.approx(5.0)
        assert r.averages["bonds"] == pytest.approx(17 / 3)
        assert r.averages["cash"] == pytest.approx(2.0)

    def test_average_on_result(self):
        r = laplace(OUTCOMES).solve()
        assert r.average == pytest.approx(17 / 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            laplace({})

    def test_single_choice(self):
        r = laplace({"only": {"good": 10, "bad": -2}}).solve()
        assert r.choice == "only"
        assert r.average == pytest.approx(4.0)

    def test_repr_marks_chosen(self):
        assert "← chosen" in repr(laplace(OUTCOMES).solve())


PROBS = {"recession": 0.2, "stagnation": 0.5, "growth": 0.3}


class TestExpectedValue:
    def test_choice(self):
        # stocks: 0.2*(-20) + 0.5*5 + 0.3*30 = -4 + 2.5 + 9 = 7.5
        # bonds:  0.2*5    + 0.5*5 + 0.3*7  = 1 + 2.5 + 2.1 = 5.6
        # cash:   0.2*2    + 0.5*2 + 0.3*2  = 2.0
        assert expected_value(OUTCOMES, PROBS).solve().choice == "stocks"

    def test_expected_values(self):
        r = expected_value(OUTCOMES, PROBS).solve()
        assert r.expected_values["stocks"] == pytest.approx(7.5)
        assert r.expected_values["bonds"] == pytest.approx(5.6)
        assert r.expected_values["cash"] == pytest.approx(2.0)

    def test_expected_on_result(self):
        r = expected_value(OUTCOMES, PROBS).solve()
        assert r.expected == pytest.approx(7.5)

    def test_probabilities_stored_on_result(self):
        r = expected_value(OUTCOMES, PROBS).solve()
        assert r.probabilities == PROBS

    def test_equal_probs_matches_laplace(self):
        equal = {s: 1 / 3 for s in ["recession", "stagnation", "growth"]}
        ev = expected_value(OUTCOMES, equal).solve()
        lap = laplace(OUTCOMES).solve()
        assert ev.choice == lap.choice
        for c in OUTCOMES:
            assert ev.expected_values[c] == pytest.approx(lap.averages[c])

    def test_empty_outcomes_raises(self):
        with pytest.raises(ValueError):
            expected_value({}, PROBS)

    def test_empty_probabilities_raises(self):
        with pytest.raises(ValueError):
            expected_value(OUTCOMES, {})

    def test_missing_scenario_raises(self):
        with pytest.raises(ValueError):
            expected_value(OUTCOMES, {"recession": 0.5, "stagnation": 0.5})

    def test_probabilities_not_summing_to_one_raises(self):
        with pytest.raises(ValueError):
            expected_value(OUTCOMES, {"recession": 0.2, "stagnation": 0.5, "growth": 0.5})

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError):
            expected_value(OUTCOMES, {"recession": -0.1, "stagnation": 0.8, "growth": 0.3})

    def test_repr_marks_chosen(self):
        assert "← chosen" in repr(expected_value(OUTCOMES, PROBS).solve())

    def test_single_choice(self):
        r = expected_value({"only": {"good": 10, "bad": -2}}, {"good": 0.7, "bad": 0.3}).solve()
        assert r.choice == "only"
        assert r.expected == pytest.approx(0.7 * 10 + 0.3 * (-2))


FLOAT_OUTCOMES = {
    "a": {"s1": -1.5, "s2": 2.5, "s3": 10.0},
    "b": {"s1": 0.5, "s2": 0.5, "s3": 1.0},
}


class TestFloatPayoffs:
    def test_maximin_float(self):
        r = maximin(FLOAT_OUTCOMES).solve()
        assert r.choice == "b"
        assert r.guaranteed == pytest.approx(0.5)

    def test_maximax_float(self):
        r = maximax(FLOAT_OUTCOMES).solve()
        assert r.choice == "a"
        assert r.best_case == pytest.approx(10.0)

    def test_minimax_regret_float(self):
        r = minimax_regret(FLOAT_OUTCOMES).solve()
        # best per scenario: s1=0.5, s2=2.5, s3=10.0
        # regrets a: s1=2.0, s2=0.0, s3=0.0 → max=2.0
        # regrets b: s1=0.0, s2=2.0, s3=9.0 → max=9.0
        assert r.choice == "a"
        assert r.max_regrets["a"] == pytest.approx(2.0)
        assert r.max_regrets["b"] == pytest.approx(9.0)

    def test_hurwicz_float(self):
        r = hurwicz(FLOAT_OUTCOMES, alpha=0.5).solve()
        # a: 0.5*10.0 + 0.5*(-1.5) = 4.25
        # b: 0.5*1.0  + 0.5*0.5    = 0.75
        assert r.choice == "a"
        assert r.scores["a"] == pytest.approx(4.25)

    def test_laplace_float(self):
        r = laplace(FLOAT_OUTCOMES).solve()
        # a: (-1.5+2.5+10.0)/3 = 11/3 ≈ 3.667
        # b: (0.5+0.5+1.0)/3   = 2/3  ≈ 0.667
        assert r.choice == "a"
        assert r.averages["a"] == pytest.approx(11 / 3)
