from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ._check import RefutationCheck, RefutationReport

_RCC_SEED = 54321
_PLACEBO_SEED = 99999
_PLACEBO_TIME_SEED = 22222
_RCC_COL = "_rcc"


def _check_placebo_group(
    data: pd.DataFrame,
    group: str,
    time: str,
    outcome: str,
    original_effect: float,
    original_se: float,
) -> RefutationCheck:
    """
    Randomly permute group labels and re-run DiD.

    Under the null (no real treatment effect), the placebo DiD estimate
    should be close to zero.  A placebo estimate larger than one standard
    error of the original suggests the result may be spurious.
    """
    rng = np.random.default_rng(_PLACEBO_SEED)

    augmented = data.assign(**{group: rng.permutation(data[group].values)})

    result = smf.ols(f"{outcome} ~ {group} * {time}", data=augmented).fit()
    interaction = f"{group}:{time}"
    placebo_effect = float(result.params[interaction])

    passed = abs(placebo_effect) <= original_se
    if passed:
        detail = (
            f"placebo estimate = {placebo_effect:.4f}  (≤ 1 SE = {original_se:.4f})  "
            f"Permuting group labels yields near-zero effect, as expected."
        )
    else:
        detail = (
            f"placebo estimate = {placebo_effect:.4f}  (> 1 SE = {original_se:.4f})  "
            f"A randomly permuted group produced a large effect — the original "
            f"result may be spurious or driven by chance differences between groups."
        )
    return RefutationCheck(name="Placebo group", passed=passed, detail=detail)


def _check_placebo_time(
    data: pd.DataFrame,
    group: str,
    time: str,
    outcome: str,
    original_effect: float,
    original_se: float,
) -> RefutationCheck:
    """
    Randomly permute time labels and re-run DiD.

    If the treatment effect is genuinely concentrated in the post period,
    scrambling which observations are labelled pre vs post should destroy
    the interaction signal and yield a placebo estimate near zero.  A large
    estimate suggests the result may be driven by something other than the
    timing of treatment.
    """
    rng = np.random.default_rng(_PLACEBO_TIME_SEED)

    augmented = data.assign(**{time: rng.permutation(data[time].values)})

    result = smf.ols(f"{outcome} ~ {group} * {time}", data=augmented).fit()
    interaction = f"{group}:{time}"
    placebo_effect = float(result.params[interaction])

    passed = abs(placebo_effect) <= original_se
    if passed:
        detail = (
            f"placebo estimate = {placebo_effect:.4f}  (≤ 1 SE = {original_se:.4f})  "
            f"Permuting time labels yields near-zero effect, as expected."
        )
    else:
        detail = (
            f"placebo estimate = {placebo_effect:.4f}  (> 1 SE = {original_se:.4f})  "
            f"A randomly permuted time produced a large effect — the original "
            f"result may not be specific to the treatment period."
        )
    return RefutationCheck(name="Placebo time", passed=passed, detail=detail)


def _check_random_common_cause(
    data: pd.DataFrame,
    group: str,
    time: str,
    outcome: str,
    original_effect: float,
    original_se: float,
) -> RefutationCheck:
    """
    Add a random noise column as an extra covariate and re-run DiD.

    Because the column is pure noise (orthogonal to group, time, and outcome),
    the interaction coefficient should not move by more than one standard error.
    A larger shift indicates the estimate is sensitive to the model specification.
    """
    rng = np.random.default_rng(_RCC_SEED)

    col = _RCC_COL
    while col in data.columns:
        col = "_" + col

    augmented = data.assign(**{col: rng.normal(size=len(data))})

    result = smf.ols(f"{outcome} ~ {group} * {time} + {col}", data=augmented).fit()
    interaction = f"{group}:{time}"
    new_effect = float(result.params[interaction])

    shift = abs(new_effect - original_effect)
    passed = shift <= original_se

    if passed:
        detail = f"estimate shifted by {shift:.4f}  (≤ 1 SE = {original_se:.4f})"
    else:
        detail = (
            f"estimate shifted by {shift:.4f}  (> 1 SE = {original_se:.4f})  "
            f"Adding a random common cause destabilised the DiD estimate."
        )
    return RefutationCheck(name="Random common cause", passed=passed, detail=detail)


class DiDRefutationReport(RefutationReport):
    """
    Results of refutation checks run against a DiD estimation.

    Obtain via ``DiDResult.refute(data)``.

    Example::

        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)
        report = result.refute(df)
        print(report.summary())
    """

    def __init__(
        self,
        checks: list[RefutationCheck],
        group: str,
        time: str,
        outcome: str,
    ) -> None:
        super().__init__(checks, treatment=group, outcome=outcome)
        self._time = time

    def _header_lines(self) -> list[str]:
        return [f"DiD Refutation Report: {self._treatment} \u00d7 {self._time} \u2192 {self._outcome}"]
