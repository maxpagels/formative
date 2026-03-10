from dataclasses import dataclass


@dataclass
class ExpectedValueResult:
    """Result of applying the expected value decision rule.

    Attributes
    ----------
    choice : str
        The choice with the highest expected payoff.
    expected : float
        Expected payoff of the chosen choice.
    expected_values : dict
        Expected payoff for every choice: {choice: float}.
    probabilities : dict
        Scenario probabilities used: {scenario: float}.
    """

    choice: str
    expected: float
    expected_values: dict
    probabilities: dict

    def __repr__(self):
        w = max(len(c) for c in self.expected_values)
        lines = ["ExpectedValueResult("]
        for c, v in self.expected_values.items():
            marker = "  ← chosen" if c == self.choice else ""
            lines.append(f"  {c:{w}s}  E[payoff]: {v:+.4g}{marker}")
        lines.append(")")
        return "\n".join(lines)


class ExpectedValue:
    """Expected value decision rule with explicit scenario probabilities.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.
    probabilities : dict
        Mapping of ``{scenario: probability}``. Must cover every scenario
        present in *outcomes* and sum to 1.0 (within 1e-6 tolerance).

    Examples
    --------
    >>> result = expected_value({
    ...     "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
    ...     "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
    ...     "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
    ... }, probabilities={"recession": 0.2, "stagnation": 0.5, "growth": 0.3}).solve()
    >>> result.choice
    'stocks'
    """

    def __init__(self, outcomes, probabilities):
        if not outcomes:
            raise ValueError("outcomes must not be empty")
        if not probabilities:
            raise ValueError("probabilities must not be empty")

        all_scenarios = {s for payoffs in outcomes.values() for s in payoffs}
        missing = all_scenarios - probabilities.keys()
        if missing:
            raise ValueError(f"probabilities missing scenarios: {missing}")

        if any(p < 0 for p in probabilities.values()):
            raise ValueError("all probabilities must be non-negative")

        total = sum(probabilities[s] for s in all_scenarios)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1.0, got {total}")

        self._outcomes = outcomes
        self._probabilities = probabilities

    def solve(self):
        """Apply the expected value rule and return the choice with the highest expected payoff.

        Returns
        -------
        ExpectedValueResult
        """
        ev = {
            choice: sum(payoff * self._probabilities[scenario] for scenario, payoff in payoffs.items())
            for choice, payoffs in self._outcomes.items()
        }
        best = max(ev, key=ev.__getitem__)
        return ExpectedValueResult(
            choice=best,
            expected=ev[best],
            expected_values=ev,
            probabilities=self._probabilities,
        )


def expected_value(outcomes, probabilities):
    """Find the choice with the highest expected payoff given explicit scenario probabilities.

    Unlike the Laplace rule, this does not assume equal probability for each
    scenario — the caller supplies their own beliefs via *probabilities*.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.
    probabilities : dict
        Mapping of ``{scenario: probability}``. Must cover every scenario
        present in *outcomes* and sum to 1.0 (within 1e-6 tolerance).

    Returns
    -------
    ExpectedValue
        Call ``.solve()`` to get the result.
    """
    return ExpectedValue(outcomes, probabilities)
