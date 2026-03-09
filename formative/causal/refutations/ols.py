from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from ._check import RefutationCheck, RefutationReport, _add_random_column


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
    augmented, col = _add_random_column(data)

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


class OLSRefutationReport(RefutationReport):
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

    def _header_lines(self) -> list[str]:
        return [f"OLS Refutation Report: {self._treatment} → {self._outcome}"]
