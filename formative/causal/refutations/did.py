from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ._check import RefutationCheck, RefutationReport, _add_random_column, _placebo_check, _shift_check

_PLACEBO_SEED = 99999
_PLACEBO_TIME_SEED = 22222


def _placebo_permutation(
    data: pd.DataFrame,
    group: str,
    time: str,
    outcome: str,
    permute_var: str,
    seed: int,
    original_se: float,
    check_name: str,
    axis_label: str,
    fail_suffix: str,
) -> RefutationCheck:
    rng = np.random.default_rng(seed)
    augmented = data.assign(**{permute_var: rng.permutation(data[permute_var].values)})
    placebo_effect = float(smf.ols(f"{outcome} ~ {group} * {time}", data=augmented).fit().params[f"{group}:{time}"])
    return _placebo_check(
        name=check_name,
        label="placebo estimate",
        placebo_effect=placebo_effect,
        original_se=original_se,
        pass_suffix=f"Permuting {axis_label} labels yields near-zero effect, as expected.",
        fail_suffix=fail_suffix,
    )


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
    return _placebo_permutation(
        data,
        group,
        time,
        outcome,
        group,
        _PLACEBO_SEED,
        original_se,
        check_name="Placebo group",
        axis_label="group",
        fail_suffix=(
            "A randomly permuted group produced a large effect — the original "
            "result may be spurious or driven by chance differences between groups."
        ),
    )


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
    return _placebo_permutation(
        data,
        group,
        time,
        outcome,
        time,
        _PLACEBO_TIME_SEED,
        original_se,
        check_name="Placebo time",
        axis_label="time",
        fail_suffix=(
            "A randomly permuted time produced a large effect — the original "
            "result may not be specific to the treatment period."
        ),
    )


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
    augmented, col = _add_random_column(data)

    result = smf.ols(f"{outcome} ~ {group} * {time} + {col}", data=augmented).fit()
    new_effect = float(result.params[f"{group}:{time}"])

    return _shift_check(
        new_effect, original_effect, original_se, "Adding a random common cause destabilised the DiD estimate."
    )


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
