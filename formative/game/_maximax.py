from dataclasses import dataclass


@dataclass
class MaximaxResult:
    """Result of applying the maximax decision rule.

    Attributes
    ----------
    choice : str
        The choice with the best best-case payoff.
    best_case : float
        The best possible payoff for the chosen choice.
    best_cases : dict
        Best-case payoff for every choice: {choice: float}.
    """

    choice: str
    best_case: float
    best_cases: dict

    def __repr__(self):
        w = max(len(c) for c in self.best_cases)
        lines = ["MaximaxResult("]
        for c, v in self.best_cases.items():
            marker = "  ← chosen" if c == self.choice else ""
            lines.append(f"  {c:{w}s}  best case: {v:+.4g}{marker}")
        lines.append(")")
        return "\n".join(lines)


class Maximax:
    """Maximax decision rule.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.

    Examples
    --------
    >>> result = maximax({
    ...     "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
    ...     "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
    ...     "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
    ... }).solve()
    >>> result.choice
    'stocks'
    """

    def __init__(self, outcomes):
        if not outcomes:
            raise ValueError("outcomes must not be empty")
        self._outcomes = outcomes

    def solve(self):
        """Apply the maximax rule and return the most optimistic choice.

        Returns
        -------
        MaximaxResult
        """
        best_cases = {choice: max(payoffs.values()) for choice, payoffs in self._outcomes.items()}
        best = max(best_cases, key=best_cases.__getitem__)
        return MaximaxResult(
            choice=best,
            best_case=best_cases[best],
            best_cases=best_cases,
        )


def maximax(outcomes):
    """Find the choice with the best best-case payoff across all scenarios.

    The optimistic counterpart to maximin: assumes the best possible scenario
    will occur and picks accordingly.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.

    Returns
    -------
    Maximax
        Call ``.solve()`` to get the result.
    """
    return Maximax(outcomes)
