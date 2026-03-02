from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ._check import RefutationCheck

_RCC_SEED = 54321
_RCC_COL = "_rcc"


def _check_random_common_cause(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: set[str],
    original_effect: float,
    original_se: float,
) -> RefutationCheck:
    """
    Add a random noise column as an extra control and re-run OLS.

    Because the column is pure noise (orthogonal to treatment and outcome),
    the estimate should not move by more than one standard error. A larger
    shift indicates the estimate is sensitive to spurious controls.
    """
    rng = np.random.default_rng(_RCC_SEED)

    col = _RCC_COL
    while col in data.columns:
        col = "_" + col

    augmented = data.assign(**{col: rng.normal(size=len(data))})

    controls = sorted(adjustment_set | {col})
    rhs = " + ".join([treatment] + controls)
    new_effect = float(smf.ols(f"{outcome} ~ {rhs}", data=augmented).fit().params[treatment])

    shift = abs(new_effect - original_effect)
    passed = shift <= original_se

    if passed:
        detail = f"estimate shifted by {shift:.4f}  (≤ 1 SE = {original_se:.4f})"
    else:
        detail = (
            f"estimate shifted by {shift:.4f}  (> 1 SE = {original_se:.4f})  "
            f"Adding a random common cause destabilised the estimate."
        )
    return RefutationCheck(name="Random common cause", passed=passed, detail=detail)


class OLSRefutationReport:
    """
    Results of refutation checks run against an OLS estimation.

    Obtain via ``OLSResult.refute(data)``.

    Example::

        result = OLSObservational(
            dag, treatment="education", outcome="income"
        ).fit(df)
        report = result.refute(df)
        print(report.summary())
    """

    def __init__(
        self,
        checks: list[RefutationCheck],
        treatment: str,
        outcome: str,
    ) -> None:
        self._checks = checks
        self._treatment = treatment
        self._outcome = outcome

    @property
    def checks(self) -> list[RefutationCheck]:
        """All checks, in the order they were run."""
        return list(self._checks)

    @property
    def passed(self) -> bool:
        """``True`` if every check passed."""
        return all(c.passed for c in self._checks)

    @property
    def failed_checks(self) -> list[RefutationCheck]:
        """Only the checks that did not pass."""
        return [c for c in self._checks if not c.passed]

    def summary(self) -> str:
        lines = [
            "",
            f"OLS Refutation Report: {self._treatment} → {self._outcome}",
            "─" * 50,
        ]
        for check in self._checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{status}]  {check.name}: {check.detail}")
        lines.append("")
        if self.passed:
            lines.append("  All checks passed.")
        else:
            n = len(self.failed_checks)
            lines.append(f"  {n} check(s) failed — see above.")
        lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
