import pytest

from formative.game import maximax, maximin, minimax_regret

OUTCOMES = {
    "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
    "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
    "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
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
        assert r.max_regrets["stocks"] == 25   # recession: 5 - (-20)
        assert r.max_regrets["bonds"] == 23    # growth: 30 - 7
        assert r.max_regrets["cash"] == 28     # growth: 30 - 2

    def test_regret_table(self):
        r = minimax_regret(OUTCOMES).solve()
        assert r.regret_table["stocks"]["recession"] == 25  # 5 - (-20)
        assert r.regret_table["bonds"]["recession"] == 0    # best in recession
        assert r.regret_table["stocks"]["growth"] == 0      # best in growth

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            minimax_regret({})

    def test_repr_marks_chosen(self):
        assert "← chosen" in repr(minimax_regret(OUTCOMES).solve())

    def test_single_choice_zero_regret(self):
        r = minimax_regret({"only": {"good": 10, "bad": -1}}).solve()
        assert r.choice == "only"
        assert r.max_regret == 0
