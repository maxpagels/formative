from __future__ import annotations

import numpy as np
import pandas as pd

from ._check import RefutationCheck
from ..estimators.matching import _propensity_scores, _att_from_ps

_RCC_SEED     = 54321
_PLACEBO_SEED = 99999
_RCC_COL      = "_rcc"


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

    passed = abs(placebo_att) <= original_se
    if passed:
        detail = (
            f"placebo ATT = {placebo_att:.4f}  (≤ 1 SE = {original_se:.4f})  "
            f"Permuting treatment labels yields near-zero effect, as expected."
        )
    else:
        detail = (
            f"placebo ATT = {placebo_att:.4f}  (> 1 SE = {original_se:.4f})  "
            f"A randomly permuted treatment produced a large effect — the original "
            f"result may be driven by residual confounding or overfitting."
        )
    return RefutationCheck(name="Placebo treatment", passed=passed, detail=detail)


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
    rng = np.random.default_rng(_RCC_SEED)

    col = _RCC_COL
    while col in data.columns:
        col = "_" + col

    augmented = data.assign(**{col: rng.normal(size=len(data))})

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

    shift = abs(new_att - original_att)
    passed = shift <= original_se

    if passed:
        detail = f"estimate shifted by {shift:.4f}  (≤ 1 SE = {original_se:.4f})"
    else:
        detail = (
            f"estimate shifted by {shift:.4f}  (> 1 SE = {original_se:.4f})  "
            f"Adding a random common cause destabilised the ATT estimate."
        )
    return RefutationCheck(name="Random common cause", passed=passed, detail=detail)


class MatchingRefutationReport:
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
            f"PSM Refutation Report: {self._treatment} → {self._outcome}",
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
