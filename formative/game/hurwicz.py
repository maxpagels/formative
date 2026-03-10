from dataclasses import dataclass


@dataclass
class HurwiczResult:
    """Result of applying the Hurwicz decision rule.

    Attributes
    ----------
    choice : str
        The choice with the highest Hurwicz score.
    score : float
        Hurwicz score of the chosen choice.
    alpha : float
        The optimism coefficient used (0 = pure maximin, 1 = pure maximax).
    scores : dict
        Hurwicz score for every choice: {choice: float}.
    """

    choice: str
    score: float
    alpha: float
    scores: dict

    def __repr__(self):
        w = max(len(c) for c in self.scores)
        lines = [f"HurwiczResult(alpha={self.alpha:.4g},"]
        for c, v in self.scores.items():
            marker = "  ← chosen" if c == self.choice else ""
            lines.append(f"  {c:{w}s}  score: {v:+.4g}{marker}")
        lines.append(")")
        return "\n".join(lines)


class Hurwicz:
    """Hurwicz decision rule.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.
    alpha : float
        Optimism coefficient in [0, 1]. At alpha=0 the rule reduces to maximin
        (pessimistic); at alpha=1 it reduces to maximax (optimistic).

    Examples
    --------
    >>> result = hurwicz({
    ...     "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
    ...     "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
    ...     "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
    ... }, alpha=0.5).solve()
    >>> result.choice
    'bonds'
    """

    def __init__(self, outcomes, alpha):
        if not outcomes:
            raise ValueError("outcomes must not be empty")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self._outcomes = outcomes
        self._alpha = alpha

    def solve(self):
        """Apply the Hurwicz rule and return the chosen action.

        Returns
        -------
        HurwiczResult
        """
        scores = {
            choice: self._alpha * max(payoffs.values()) + (1 - self._alpha) * min(payoffs.values())
            for choice, payoffs in self._outcomes.items()
        }
        best = max(scores, key=scores.__getitem__)
        return HurwiczResult(choice=best, score=scores[best], alpha=self._alpha, scores=scores)


def hurwicz(outcomes, alpha):
    """Find the choice with the highest weighted combination of best and worst payoffs.

    The Hurwicz criterion scores each choice as:
    ``alpha * best_case + (1 - alpha) * worst_case``

    At ``alpha=0`` this is identical to maximin; at ``alpha=1`` it is identical
    to maximax. Intermediate values express a blend of optimism and pessimism.

    Parameters
    ----------
    outcomes : dict of dict
        Mapping of ``{choice: {scenario: payoff}}``.
    alpha : float
        Optimism coefficient in [0, 1].

    Returns
    -------
    Hurwicz
        Call ``.solve()`` to get the result.
    """
    return Hurwicz(outcomes, alpha)
