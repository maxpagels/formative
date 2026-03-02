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
    original_effect: float,
    original_se: float,
) -> RefutationCheck:
    """
    Add a random noise column as an extra control and re-run OLS.

    Under randomisation, the ATE is independent of any additional covariate,
    so the estimate should not move by more than one standard error. A larger
    shift suggests something other than randomisation is driving the result.
    """
    rng = np.random.default_rng(_RCC_SEED)

    col = _RCC_COL
    while col in data.columns:
        col = "_" + col

    augmented = data.copy()
    augmented[col] = rng.normal(size=len(data))

    new_effect = float(
        smf.ols(f"{outcome} ~ {treatment} + {col}", data=augmented).fit().params[treatment]
    )

    shift = abs(new_effect - original_effect)
    passed = shift <= original_se

    if passed:
        detail = f"estimate shifted by {shift:.4f}  (≤ 1 SE = {original_se:.4f})"
    else:
        detail = (
            f"estimate shifted by {shift:.4f}  (> 1 SE = {original_se:.4f})  "
            f"Adding a random covariate destabilised the estimate — "
            f"verify that treatment was truly randomised."
        )
    return RefutationCheck(name="Random common cause", passed=passed, detail=detail)


class RCTRefutationReport:
    """
    Results of refutation checks run against an RCT estimation.

    Obtain via ``RCTResult.refute(data)``.

    Example::

        result = RCT(dag, treatment="treatment", outcome="outcome").fit(df)
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
            f"RCT Refutation Report: {self._treatment} → {self._outcome}",
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
