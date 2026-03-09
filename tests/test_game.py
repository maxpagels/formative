import pytest

from formative.game import maximin


class TestMaximin:
    def test_bonds_stocks_cash(self):
        result = maximin({
            "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
            "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
            "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
        }).solve()
        assert result.choice == "bonds"
        assert result.guaranteed == 5

    def test_worst_cases_correct(self):
        result = maximin({
            "A": {"x": 3, "y": 1},
            "B": {"x": 0, "y": 4},
        }).solve()
        assert result.worst_cases["A"] == 1
        assert result.worst_cases["B"] == 0
        assert result.choice == "A"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            maximin({})

    def test_single_choice(self):
        result = maximin({"only": {"good": 10, "bad": -1}}).solve()
        assert result.choice == "only"
        assert result.guaranteed == -1

    def test_repr_marks_chosen(self):
        result = maximin({"A": {"x": 1}, "B": {"x": 2}}).solve()
        assert "← chosen" in repr(result)

    def test_all_worst_cases_present(self):
        outcomes = {
            "stocks": {"recession": -20, "stagnation": 5,  "growth": 30},
            "bonds":  {"recession":   5, "stagnation": 5,  "growth":  7},
            "cash":   {"recession":   2, "stagnation": 2,  "growth":  2},
        }
        result = maximin(outcomes).solve()
        assert set(result.worst_cases.keys()) == set(outcomes.keys())
        assert result.worst_cases["stocks"] == -20
        assert result.worst_cases["bonds"] == 5
        assert result.worst_cases["cash"] == 2
