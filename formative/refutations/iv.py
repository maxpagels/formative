from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS as _IV2SLS

from ._check import RefutationCheck

_FIRST_STAGE_F_THRESHOLD = 10.0
_RCC_SEED = 54321
_RCC_COL = "_rcc"


def _check_first_stage_f(
    data: pd.DataFrame,
    treatment: str,
    instrument: str,
    controls: list[str],
    threshold: float = _FIRST_STAGE_F_THRESHOLD,
) -> RefutationCheck:
    """
    Refit the first-stage regression and compute the partial F-statistic
    for the instrument (``H0: instrument coefficient = 0``).

    Uses the partial F rather than the overall model F so the test
    remains meaningful when observed confounders are included as controls.
    Conventional threshold: F < 10 indicates a weak instrument
    (Stock & Yogo, 2005).
    """
    rhs = " + ".join([instrument] + controls)
    first_stage = smf.ols(f"{treatment} ~ {rhs}", data=data).fit()
    f_stat = float(first_stage.f_test(f"{instrument} = 0").fvalue)

    passed = f_stat >= threshold
    if passed:
        detail = f"F = {f_stat:.2f}  (threshold: F ≥ {threshold:.0f})"
    else:
        detail = (
            f"F = {f_stat:.2f}  (threshold: F ≥ {threshold:.0f})  "
            f"Weak instrument detected — the instrument explains little "
            f"variation in treatment. IV estimates may be severely biased "
            f"and confidence intervals unreliable."
        )
    return RefutationCheck(name="First-stage F-statistic", passed=passed, detail=detail)


def _check_random_common_cause(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    instrument: str,
    adjustment_set: set[str],
    original_effect: float,
    original_se: float,
) -> RefutationCheck:
    """
    Add a random noise column as an extra control and re-run 2SLS.

    Because the column is pure noise (orthogonal to instrument, treatment, and
    outcome), the IV estimate should not move by more than one standard error.
    A larger shift indicates the estimate is sensitive to spurious controls.
    """
    rng = np.random.default_rng(_RCC_SEED)

    col = _RCC_COL
    while col in data.columns:
        col = "_" + col

    augmented = data.copy()
    augmented[col] = rng.normal(size=len(data))

    controls = sorted(adjustment_set | {col})
    X = sm.add_constant(augmented[[treatment] + controls], prepend=True)
    Z_mat = sm.add_constant(augmented[[instrument] + controls], prepend=True)
    Z_mat.columns = X.columns

    new_effect = float(
        _IV2SLS(endog=augmented[outcome], exog=X, instrument=Z_mat).fit().params[treatment]
    )

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


class IVRefutationReport:
    """
    Results of refutation checks run against an IV (2SLS) estimation.

    Obtain via ``IVResult.refute(data)``. Each check is a ``RefutationCheck``
    in ``.checks``. The overall verdict is ``.passed``.

    Example::

        result = IV2SLS(
            dag, treatment="education", outcome="income", instrument="proximity"
        ).fit(df)
        report = result.refute(df)
        print(report.summary())
    """

    def __init__(
        self,
        checks: list[RefutationCheck],
        treatment: str,
        outcome: str,
        instrument: str,
    ) -> None:
        self._checks = checks
        self._treatment = treatment
        self._outcome = outcome
        self._instrument = instrument

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
            f"IV Refutation Report: {self._treatment} → {self._outcome}",
            f"  Instrument: {self._instrument}",
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
