from dataclasses import dataclass, field


@dataclass
class MinimaxRegretResult:
    """Result of applying the minimax regret decision rule.

    Regret for a given choice and scenario is the difference between the best
    payoff available in that scenario and the payoff actually received. The
    minimax regret rule picks the choice whose worst-case regret is smallest.

    Attributes
    ----------
    choice : str
        The choice that minimises the maximum regret.
    max_regret : float
        The maximum regret for the chosen choice.
    max_regrets : dict
        Maximum regret for every choice: {choice: float}.
    regret_table : dict of dict
        Full regret for every (choice, scenario) pair.
    """

    choice: str
    max_regret: float
    max_regrets: dict
    regret_table: dict = field(repr=False)

    def __repr__(self):
        w = max(len(c) for c in self.max_regrets)
        lines = ["MinimaxRegretResult("]
        for c, v in self.max_regrets.items():
            marker = "  ← chosen" if c == self.choice else ""
            lines.append(f"  {c:{w}s}  max regret: {v:+.4g}{marker}")
        lines.append(")")
        return "\n".join(lines)


class MinimaxRegret:
    """Minimax regret decision rule.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.

    Examples
    --------
    >>> result = minimax_regret({
    ...     "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
    ...     "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
    ...     "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
    ... }).solve()
    >>> result.choice
    'bonds'
    """

    def __init__(self, outcomes):
        if not outcomes:
            raise ValueError("outcomes must not be empty")
        self._outcomes = outcomes

    def solve(self):
        """Apply the minimax regret rule.

        Returns
        -------
        MinimaxRegretResult
        """
        choices = list(self._outcomes.keys())
        scenarios = list(next(iter(self._outcomes.values())).keys())

        # Best available payoff in each scenario across all choices
        best_in_scenario = {
            s: max(self._outcomes[c][s] for c in choices)
            for s in scenarios
        }

        # Regret: how much you miss out on by not making the best choice
        regret_table = {
            c: {s: best_in_scenario[s] - self._outcomes[c][s] for s in scenarios}
            for c in choices
        }

        max_regrets = {c: max(regret_table[c].values()) for c in choices}
        best = min(max_regrets, key=max_regrets.__getitem__)

        return MinimaxRegretResult(
            choice=best,
            max_regret=max_regrets[best],
            max_regrets=max_regrets,
            regret_table=regret_table,
        )


def minimax_regret(outcomes):
    """Find the choice that minimises the worst-case regret across all scenarios.

    Regret for a choice in a given scenario is the gap between the best payoff
    available in that scenario and what you actually receive. This rule picks
    the choice where that gap is smallest in the worst case.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.

    Returns
    -------
    MinimaxRegret
        Call ``.solve()`` to get the result.
    """
    return MinimaxRegret(outcomes)
