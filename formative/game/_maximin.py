from dataclasses import dataclass


@dataclass
class MaximinResult:
    """Result of applying the maximin decision rule.

    Attributes
    ----------
    choice : str
        The choice that maximises the worst-case payoff.
    guaranteed : float
        The payoff guaranteed by that choice, no matter which scenario occurs.
    worst_cases : dict
        Worst-case payoff for every choice: {choice: float}.
    """

    choice: str
    guaranteed: float
    worst_cases: dict

    def __repr__(self):
        w = max(len(c) for c in self.worst_cases)
        lines = ["MaximinResult("]
        for c, v in self.worst_cases.items():
            marker = "  ← chosen" if c == self.choice else ""
            lines.append(f"  {c:{w}s}  worst case: {v:+.4g}{marker}")
        lines.append(")")
        return "\n".join(lines)


class Maximin:
    """Maximin decision rule.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.

    Examples
    --------
    >>> result = maximin({
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
        """Apply the maximin rule and return the safest choice.

        Returns
        -------
        MaximinResult
        """
        worst_cases = {
            choice: min(payoffs.values())
            for choice, payoffs in self._outcomes.items()
        }
        best = max(worst_cases, key=worst_cases.__getitem__)
        return MaximinResult(
            choice=best,
            guaranteed=worst_cases[best],
            worst_cases=worst_cases,
        )


def maximin(outcomes):
    """Find the choice that maximises the worst-case payoff across all scenarios.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.

    Returns
    -------
    Maximin
        Call ``.solve()`` to get the result.

    Examples
    --------
    >>> maximin({
    ...     "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
    ...     "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
    ...     "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
    ... }).solve()
    """
    return Maximin(outcomes)
