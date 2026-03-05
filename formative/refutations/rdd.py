from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from ._check import RefutationCheck, RefutationReport, _add_random_column

_PLACEBO_SEED = 99999


def _check_placebo_cutoff(
    data: pd.DataFrame,
    running_var: str,
    treatment: str,
    outcome: str,
    cutoff: float,
) -> RefutationCheck:
    """
    Fit an RDD at a false cutoff placed in the all-control region.

    All observations strictly below the true cutoff were never treated,
    so fitting a sharp RDD at the median of that region should produce
    an estimate near zero.  The check passes when the placebo effect's
    99% confidence interval includes zero (|placebo| ≤ 2.58 × its own SE),
    i.e. the placebo is not statistically distinguishable from zero at the
    1% level.  A placebo that clears this bar suggests the outcome is not
    smooth at the true cutoff or the original result is spurious.
    """
    below_mask = data[running_var] < cutoff
    placebo_cutoff = float(data.loc[below_mask, running_var].median())

    sub = data.loc[below_mask].copy()
    sub["_rdd_t"] = (sub[running_var] >= placebo_cutoff).astype(float)
    sub["_rdd_r"] = sub[running_var] - placebo_cutoff

    result = smf.ols(f"{outcome} ~ _rdd_t + _rdd_r + _rdd_t:_rdd_r", data=sub).fit()
    placebo_effect = float(result.params["_rdd_t"])
    placebo_se = float(result.bse["_rdd_t"])
    threshold = 2.58 * placebo_se  # 99% CI half-width

    passed = abs(placebo_effect) <= threshold
    if passed:
        detail = (
            f"placebo estimate = {placebo_effect:.4f}  "
            f"(99% CI half-width = {threshold:.4f})  "
            f"Fitting a false cutoff in the control region yields a near-zero effect, as expected."
        )
    else:
        detail = (
            f"placebo estimate = {placebo_effect:.4f}  "
            f"(> 99% CI half-width = {threshold:.4f})  "
            f"A false cutoff in the control region produced a statistically significant effect — "
            f"the original result may be spurious or the outcome is not smooth at the true cutoff."
        )
    return RefutationCheck(name="Placebo cutoff", passed=passed, detail=detail)


def _check_random_common_cause(
    data: pd.DataFrame,
    running_var: str,
    treatment: str,
    outcome: str,
    cutoff: float,
    original_effect: float,
    original_se: float,
) -> RefutationCheck:
    """
    Add a random noise column as an extra covariate and re-run the RDD.

    Because the column is pure noise (orthogonal to the running variable
    and outcome), the LATE at the cutoff estimate should not move by more than one
    standard error.  A larger shift indicates sensitivity to model
    specification.
    """
    augmented, col = _add_random_column(data)
    augmented = augmented.assign(**{
        "_rdd_r": augmented[running_var] - cutoff,
        treatment: (augmented[running_var] >= cutoff).astype(float),
    })

    result = smf.ols(
        f"{outcome} ~ {treatment} + _rdd_r + {treatment}:_rdd_r + {col}",
        data=augmented,
    ).fit()
    new_effect = float(result.params[treatment])

    shift = abs(new_effect - original_effect)
    passed = shift <= original_se

    if passed:
        detail = f"estimate shifted by {shift:.4f}  (\u2264 1 SE = {original_se:.4f})"
    else:
        detail = (
            f"estimate shifted by {shift:.4f}  (> 1 SE = {original_se:.4f})  "
            f"Adding a random common cause destabilised the RDD estimate."
        )
    return RefutationCheck(name="Random common cause", passed=passed, detail=detail)


class RDDRefutationReport(RefutationReport):
    """
    Results of refutation checks run against an RDD estimation.

    Obtain via ``RDDResult.refute(data)``.

    Example::

        result = RDD(dag, treatment="treatment", running_var="score",
                     cutoff=0.0, outcome="outcome").fit(df)
        report = result.refute(df)
        print(report.summary())
    """

    def __init__(
        self,
        checks: list[RefutationCheck],
        treatment: str,
        running_var: str,
        cutoff: float,
        outcome: str,
    ) -> None:
        super().__init__(checks, treatment=treatment, outcome=outcome)
        self._running_var = running_var
        self._cutoff = cutoff

    def _header_lines(self) -> list[str]:
        return [
            f"RDD Refutation Report: {self._running_var} → {self._treatment} → {self._outcome}"
            f"  |  cutoff: {self._cutoff}"
        ]
