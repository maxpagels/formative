"""
Refutation checks for heterogeneous-effects (CATE) estimations.

These run instead of the plain checks when a result was fitted with an
``effect_modifier``; they refit the full interaction model via ``_fit_cate``
so the comparison is like-for-like. Reports reuse ``OLSRefutationReport`` /
``RCTRefutationReport`` — pass ``adjustment_set=set()`` for RCT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..estimators._cate import _fit_cate
from ._check import RefutationCheck, _add_random_column, _shift_check

_PLACEBO_MODIFIER_SEED = 77777
_RANDOM_MODIFIER_SEED = 31313
_HETEROGENEITY_ALPHA = 0.01
_RANDOM_MODIFIER_LEVELS = 3


def _check_random_common_cause(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifier: str,
    adjustment_set: set[str],
    original_effect: float,
    original_se: float,
) -> RefutationCheck:
    """
    Add a random noise column as an extra control and refit the interaction model.

    Because the column is pure noise (orthogonal to treatment, modifier, and
    outcome), the weighted average effect should not move by more than one
    standard error. A larger shift indicates the estimate is sensitive to
    spurious controls.
    """
    augmented, col = _add_random_column(data)

    fit = _fit_cate(augmented, treatment, outcome, modifier, adjustment_set | {col})

    return _shift_check(
        fit.effect, original_effect, original_se, "Adding a random common cause destabilised the estimate."
    )


def _check_placebo_modifier(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifier: str,
    adjustment_set: set[str],
) -> RefutationCheck:
    """
    Randomly permute the modifier column and refit the interaction model.

    Permuting the modifier destroys any real link between it and the treatment
    effect, so the placebo homogeneity test should be non-significant
    (p > 0.01). Heterogeneity that survives having its modifier scrambled is
    an artifact of the data or specification, not real effect modification.
    """
    rng = np.random.default_rng(_PLACEBO_MODIFIER_SEED)
    augmented = data.assign(**{modifier: rng.permutation(data[modifier].values)})

    try:
        fit = _fit_cate(augmented, treatment, outcome, modifier, adjustment_set)
    except Exception:
        return RefutationCheck(
            name="Placebo modifier",
            passed=False,
            detail="Interaction model failed on permuted modifier — check data quality.",
        )

    p = fit.homogeneity_pvalue
    passed = p > _HETEROGENEITY_ALPHA
    if passed:
        detail = (
            f"placebo homogeneity p = {p:.4f}  (> {_HETEROGENEITY_ALPHA})  "
            f"Permuting {modifier} labels removes the heterogeneity, as expected."
        )
    else:
        detail = (
            f"placebo homogeneity p = {p:.4f}  (≤ {_HETEROGENEITY_ALPHA})  "
            f"A randomly permuted modifier still shows significant heterogeneity — "
            f"the group differences may be spurious."
        )
    return RefutationCheck(name="Placebo modifier", passed=passed, detail=detail)


def _check_random_modifier(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifier: str,
    adjustment_set: set[str],
) -> RefutationCheck:
    """
    Interact treatment with a random discrete column instead of the modifier.

    The noise column is independent of everything, so its interaction with
    treatment should show no significant heterogeneity (p > 0.01). The real
    modifier stays in the model as a plain control. A significant result
    means the specification manufactures heterogeneity out of noise.
    """
    rng = np.random.default_rng(_RANDOM_MODIFIER_SEED)
    col = "_rmod"
    while col in data.columns:
        col = "_" + col
    augmented = data.assign(**{col: rng.integers(0, _RANDOM_MODIFIER_LEVELS, size=len(data))})

    try:
        fit = _fit_cate(augmented, treatment, outcome, col, adjustment_set | {modifier})
    except Exception:
        return RefutationCheck(
            name="Random modifier",
            passed=False,
            detail="Interaction model failed with a random modifier — check data quality.",
        )

    p = fit.homogeneity_pvalue
    passed = p > _HETEROGENEITY_ALPHA
    if passed:
        detail = f"random-modifier homogeneity p = {p:.4f}  (> {_HETEROGENEITY_ALPHA})  No spurious heterogeneity."
    else:
        detail = (
            f"random-modifier homogeneity p = {p:.4f}  (≤ {_HETEROGENEITY_ALPHA})  "
            f"A pure-noise modifier shows significant heterogeneity — group "
            f"effects from this specification are not trustworthy."
        )
    return RefutationCheck(name="Random modifier", passed=passed, detail=detail)
