from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from ._check import RefutationCheck, RefutationReport, _add_random_column, _shift_check


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
    augmented, col = _add_random_column(data)

    new_effect = float(smf.ols(f"{outcome} ~ {treatment} + {col}", data=augmented).fit().params[treatment])

    return _shift_check(
        new_effect,
        original_effect,
        original_se,
        "Adding a random covariate destabilised the estimate — verify that treatment was truly randomised.",
    )


class RCTRefutationReport(RefutationReport):
    """
    Results of refutation checks run against an RCT estimation.

    Obtain via ``RCTResult.refute(data)``.

    Example::

        result = RCT(dag, treatment="treatment", outcome="outcome").fit(df)
        report = result.refute(df)
        print(report.summary())
    """

    def _header_lines(self) -> list[str]:
        return [f"RCT Refutation Report: {self._treatment} → {self._outcome}"]
