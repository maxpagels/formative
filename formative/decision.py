from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class DecisionReport:
    """
    The result of a cost-benefit decision analysis built on a causal estimate.

    Answers: given what we estimated, should we apply the treatment?

    Attributes
    ----------
    treatment : str
        Name of the treatment variable.
    outcome : str
        Name of the outcome variable.
    cost : float
        Cost per unit of treatment applied.
    benefit : float
        Benefit (revenue, utility, etc.) per unit increase in the outcome.
    net_benefit : float
        Point-estimate net benefit: ``effect * benefit - cost``.
    net_benefit_ci : tuple[float, float]
        95% CI on net benefit, propagated from the causal estimate's CI.
    optimal : str
        ``"treat"`` if the point-estimate net benefit is positive, else ``"don't treat"``.
    robust : bool
        ``True`` if the optimal decision is the same at both CI bounds — i.e.
        the conclusion does not flip under estimation uncertainty.
    p_beneficial : float
        Probability that the true net benefit is positive, assuming the causal
        estimate is normally distributed around its point estimate.
    """

    treatment: str
    outcome: str
    cost: float
    benefit: float
    net_benefit: float
    net_benefit_ci: tuple[float, float]
    optimal: str
    robust: bool
    p_beneficial: float

    def value_of_information(self, target_confidence: float = 0.95) -> str:
        """
        Estimate how much the standard error would need to shrink — and
        approximately how many times larger the sample would need to be — to
        reach ``target_confidence`` that the optimal decision is correct.

        If the decision is already at or above ``target_confidence``, reports
        that no additional data is needed.

        Parameters
        ----------
        target_confidence : float
            Desired probability that net benefit > 0 (default 0.95).

        Returns
        -------
        str
            A human-readable description of the value of information.
        """
        if self.p_beneficial >= target_confidence:
            return (
                f"Decision confidence is already {self.p_beneficial:.1%}, "
                f"at or above the target of {target_confidence:.1%}. "
                f"No additional data is needed to be confident in this decision."
            )

        # net_benefit / (se_net * z_target) = 1  →  se_net_required = net_benefit / z_target
        # se_net = std_err_ate * benefit  →  std_err_required = se_net_required / benefit
        z_target = norm.ppf(target_confidence)
        se_net_current = abs(self.net_benefit_ci[1] - self.net_benefit_ci[0]) / (2 * 1.96)
        if se_net_current == 0:
            return "Cannot compute value of information: standard error is zero."

        se_net_required = abs(self.net_benefit) / z_target
        se_ratio = se_net_current / se_net_required
        n_multiplier = se_ratio**2

        return (
            f"Current decision confidence: {self.p_beneficial:.1%} "
            f"(target: {target_confidence:.1%}).\n"
            f"The standard error on net benefit would need to shrink by "
            f"{(1 - 1 / se_ratio):.1%} to reach the target.\n"
            f"This requires approximately {n_multiplier:.1f}x the current sample size."
        )

    def summary(self) -> str:
        """Formatted summary of the decision analysis."""
        lo, hi = self.net_benefit_ci
        robust_str = "Yes — decision is stable across 95% CI" if self.robust else "No — decision flips within 95% CI"

        def row(label: str, value: str) -> str:
            return f"  {label:<28} : {value}"

        lines = [
            "",
            f"Decision Analysis: {self.treatment} → {self.outcome}",
            "─" * 50,
            row("Cost per unit of treatment", f"{self.cost:>10.4f}"),
            row("Benefit per unit of outcome", f"{self.benefit:>10.4f}"),
            "",
            row("Net benefit (point estimate)", f"{self.net_benefit:>+10.4f}"),
            row("Net benefit 95% CI", f"[{lo:+.4f}, {hi:+.4f}]"),
            "",
            row("Optimal decision", self.optimal),
            row("Decision confidence", f"{self.p_beneficial:>10.1%}"),
            row("Robust to estimation error", robust_str),
            "",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _decide(
    effect: float,
    std_err: float,
    conf_int: tuple[float, float],
    treatment: str,
    outcome: str,
    cost: float,
    benefit: float,
) -> DecisionReport:
    """
    Compute a cost-benefit decision from a causal point estimate and its uncertainty.

    This is the internal implementation shared across all result classes.
    Call via ``result.decide(cost, benefit)`` rather than directly.

    Parameters
    ----------
    effect : float
        Causal point estimate (ATE, LATE, ATT, etc.).
    std_err : float
        Standard error of the estimate.
    conf_int : tuple[float, float]
        95% confidence interval ``(lower, upper)``.
    treatment : str
        Name of the treatment variable.
    outcome : str
        Name of the outcome variable.
    cost : float
        Cost per unit of treatment applied.
    benefit : float
        Benefit (revenue, utility, etc.) per unit increase in the outcome.

    Returns
    -------
    DecisionReport
    """
    net_benefit = effect * benefit - cost
    ci_lo = conf_int[0] * benefit - cost
    ci_hi = conf_int[1] * benefit - cost

    optimal = "treat" if net_benefit > 0 else "don't treat"
    robust = (ci_lo > 0) == (ci_hi > 0)

    se_net = std_err * abs(benefit)
    if se_net > 0:
        p_beneficial = float(norm.cdf(net_benefit / se_net))
    else:
        p_beneficial = 1.0 if net_benefit > 0 else 0.0

    return DecisionReport(
        treatment=treatment,
        outcome=outcome,
        cost=cost,
        benefit=benefit,
        net_benefit=net_benefit,
        net_benefit_ci=(ci_lo, ci_hi),
        optimal=optimal,
        robust=robust,
        p_beneficial=p_beneficial,
    )
