from __future__ import annotations

import numpy as np
import pandas as pd

from ..estimators.matching import _att_from_ps, _propensity_scores
from ._check import RefutationCheck, RefutationReport, _add_random_column, _placebo_check, _shift_check

_PLACEBO_SEED = 99999


def _check_placebo_treatment(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: set[str],
    original_att: float,
    original_se: float,
) -> RefutationCheck:
    """
    Permute treatment labels at random and re-run matching.

    Under the null (no real effect), the placebo ATT should be close to zero.
    A placebo ATT larger than one standard error of the original estimate
    suggests the original result may be spurious.
    """
    rng = np.random.default_rng(_PLACEBO_SEED)

    augmented = data.assign(**{treatment: rng.permutation(data[treatment].values)})

    try:
        ps = _propensity_scores(augmented, treatment, adjustment_set)
        placebo_att = _att_from_ps(
            augmented[treatment].values.astype(float),
            augmented[outcome].values.astype(float),
            ps,
        )
    except Exception:
        return RefutationCheck(
            name="Placebo treatment",
            passed=False,
            detail="Matching failed on permuted treatment — check data quality.",
        )

    return _placebo_check(
        name="Placebo treatment",
        label="placebo ATT",
        placebo_effect=placebo_att,
        original_se=original_se,
        pass_suffix="Permuting treatment labels yields near-zero effect, as expected.",
        fail_suffix=(
            "A randomly permuted treatment produced a large effect — the original "
            "result may be driven by residual confounding or overfitting."
        ),
    )


def _check_random_common_cause(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: set[str],
    original_att: float,
    original_se: float,
) -> RefutationCheck:
    """
    Add a random noise column as an extra covariate in the propensity score
    model and re-run matching.

    Because the column is pure noise (orthogonal to everything), the ATT should
    not move by more than one standard error. A larger shift indicates the
    estimate is sensitive to the propensity model specification.
    """
    augmented, col = _add_random_column(data)

    new_adjustment = adjustment_set | {col}

    try:
        ps = _propensity_scores(augmented, treatment, new_adjustment)
        new_att = _att_from_ps(
            augmented[treatment].values.astype(float),
            augmented[outcome].values.astype(float),
            ps,
        )
    except Exception:
        return RefutationCheck(
            name="Random common cause",
            passed=False,
            detail="Matching failed after adding random covariate — check data quality.",
        )

    return _shift_check(
        new_att, original_att, original_se, "Adding a random common cause destabilised the ATT estimate."
    )


class MatchingRefutationReport(RefutationReport):
    """
    Results of refutation checks run against a propensity score matching
    estimation.

    Obtain via ``MatchingResult.refute(data)``. Each check is a
    ``RefutationCheck`` in ``.checks``. The overall verdict is ``.passed``.

    Example::

        result = PropensityScoreMatching(
            dag, treatment="education", outcome="income"
        ).fit(df)
        report = result.refute(df)
        print(report.summary())
    """

    def _header_lines(self) -> list[str]:
        return [f"PSM Refutation Report: {self._treatment} → {self._outcome}"]
