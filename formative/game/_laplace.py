from dataclasses import dataclass


@dataclass
class LaplaceResult:
    """Result of applying the Laplace decision rule.

    Attributes
    ----------
    choice : str
        The choice with the highest average payoff.
    average : float
        Average payoff of the chosen choice.
    averages : dict
        Average payoff for every choice: {choice: float}.
    """

    choice: str
    average: float
    averages: dict

    def __repr__(self):
        w = max(len(c) for c in self.averages)
        lines = ["LaplaceResult("]
        for c, v in self.averages.items():
            marker = "  ← chosen" if c == self.choice else ""
            lines.append(f"  {c:{w}s}  average: {v:+.4g}{marker}")
        lines.append(")")
        return "\n".join(lines)


class Laplace:
    """Laplace decision rule.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.

    Examples
    --------
    >>> result = laplace({
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
        """Apply the Laplace rule and return the choice with the highest average payoff.

        Returns
        -------
        LaplaceResult
        """
        averages = {choice: sum(payoffs.values()) / len(payoffs) for choice, payoffs in self._outcomes.items()}
        best = max(averages, key=averages.__getitem__)
        return LaplaceResult(choice=best, average=averages[best], averages=averages)


def laplace(outcomes):
    """Find the choice with the highest average payoff across all scenarios.

    Treats every scenario as equally likely (Laplace's principle of indifference)
    and picks the choice with the best expected payoff under that assumption.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.

    Returns
    -------
    Laplace
        Call ``.solve()`` to get the result.
    """
    return Laplace(outcomes)
