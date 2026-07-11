from __future__ import annotations

from .._assumptions import Assumption


class _BaseResult:
    """
    Shared behaviour for estimation result classes.

    Subclasses set ``self._treatment`` and ``self._outcome`` in ``__init__``
    and define a class-level ``_ASSUMPTIONS`` list. ``decide()`` reads the
    ``effect`` / ``std_err`` / ``conf_int`` properties, which subclasses
    (or ``_StatsmodelsResult``) provide.
    """

    _ASSUMPTIONS: list[Assumption] = []
    _treatment: str
    _outcome: str

    @property
    def assumptions(self) -> list[Assumption]:
        """Modelling assumptions required for a causal interpretation."""
        return list(self._ASSUMPTIONS)

    def _assumptions_lines(self, width: int = 48) -> list[str]:
        """Assumptions footer appended to every ``summary()``."""
        lines = ["", "  Assumptions", "  " + "┄" * width]
        lines += [f"  {a.fmt_tag()}  {a.name}" for a in self._ASSUMPTIONS]
        lines.append("")
        return lines

    def _extra_summary_lines(self) -> list[str]:
        """Hook for subclasses to insert extra blocks before the assumptions footer."""
        return []

    def decide(self, cost: float, benefit: float):
        """
        Compute a cost-benefit decision analysis from this causal estimate.

        Parameters
        ----------
        cost : float
            Cost per unit of treatment applied.
        benefit : float
            Benefit (revenue, utility, etc.) per unit increase in the outcome.

        Returns
        -------
        DecisionReport
            Optimal decision, net benefit, CI, confidence, and robustness flag.
        """
        from ..decision import _decide

        return _decide(self.effect, self.std_err, self.conf_int, self._treatment, self._outcome, cost, benefit)

    def __repr__(self) -> str:
        return self.summary()


class _StatsmodelsResult(_BaseResult):
    """
    Result backed by a fitted statsmodels regression.

    Effect statistics are read from ``self._result`` at parameter
    ``self._param``, which defaults to the treatment name (DiD overrides
    it with the group × time interaction term).
    """

    @property
    def _param(self) -> str:
        return self._treatment

    @property
    def effect(self) -> float:
        """Point estimate of the causal effect of treatment on outcome."""
        return float(self._result.params[self._param])

    @property
    def std_err(self) -> float:
        """Standard error of the treatment effect estimate."""
        return float(self._result.bse[self._param])

    @property
    def conf_int(self) -> tuple[float, float]:
        """95% confidence interval for the treatment effect."""
        ci = self._result.conf_int()
        return (float(ci.loc[self._param, 0]), float(ci.loc[self._param, 1]))

    @property
    def pvalue(self) -> float:
        """p-value for the treatment effect (``H0: effect = 0``)."""
        return float(self._result.pvalues[self._param])

    @property
    def statsmodels_result(self):
        """The underlying statsmodels result, for full diagnostics."""
        return self._result
