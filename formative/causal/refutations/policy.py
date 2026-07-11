"""
Refutation checks for learned treatment policies.

Both checks re-run the full policy learner (``_fit_policy`` — cross-fitted
scores, tree search, honest value) on perturbed inputs, so the comparison is
like-for-like. They test the honest *value*, not the tree's structure: with a
depth budget, in-sample search will happily split on noise, but honest
out-of-fold value collapses to ≈ 0 when there is nothing real to target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._check import RefutationCheck, RefutationReport

_PLACEBO_POLICY_SEED = 88888
_RANDOM_POLICY_MODIFIER_SEED = 46464
_RANDOM_POLICY_LEVELS = 3


class PolicyRefutationReport(RefutationReport):
    """Refutation report for a policy learned via ``learn_policy()``."""

    def _header_lines(self) -> list[str]:
        return [f"Policy Refutation Report: {self._treatment} → {self._outcome}"]


def _check_placebo_modifiers(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifiers: list[str],
    cost: float,
    benefit: float,
    max_depth: int,
) -> RefutationCheck:
    """
    Permute every candidate feature column independently and re-learn.

    Scrambled features carry no information about who benefits, so the
    placebo policy's honest value should not be significantly positive
    (z ≤ 2). A significantly positive placebo value means the learner is
    manufacturing value out of noise.
    """
    from ..estimators.policy import _fit_policy

    rng = np.random.default_rng(_PLACEBO_POLICY_SEED)
    augmented = data.assign(**{m: rng.permutation(data[m].to_numpy()) for m in modifiers})

    try:
        fit = _fit_policy(augmented, treatment, outcome, modifiers, cost, benefit, max_depth)
    except Exception:
        return RefutationCheck(
            name="Placebo modifiers",
            passed=False,
            detail="Policy learner failed on permuted features — check data quality.",
        )

    passed = fit.value <= 2 * fit.value_se
    if passed:
        detail = f"placebo policy value = {fit.value:+.4f} (SE {fit.value_se:.4f})  Not significantly positive."
    else:
        detail = (
            f"placebo policy value = {fit.value:+.4f} (SE {fit.value_se:.4f})  "
            f"A policy learned on scrambled features still looks valuable — "
            f"the value estimate is not trustworthy."
        )
    return RefutationCheck(name="Placebo modifiers", passed=passed, detail=detail)


def _check_random_modifier(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifiers: list[str],
    cost: float,
    benefit: float,
    max_depth: int,
    original_value: float,
    original_se: float,
) -> RefutationCheck:
    """
    Add a pure-noise feature to the candidates and re-learn.

    Noise carries no targeting information, so the honest value should not
    improve by more than one standard error of the original estimate.
    """
    from ..estimators.policy import _fit_policy

    rng = np.random.default_rng(_RANDOM_POLICY_MODIFIER_SEED)
    col = "_rpol"
    while col in data.columns:
        col = "_" + col
    augmented = data.assign(**{col: rng.integers(0, _RANDOM_POLICY_LEVELS, size=len(data))})

    try:
        fit = _fit_policy(augmented, treatment, outcome, modifiers + [col], cost, benefit, max_depth)
    except Exception:
        return RefutationCheck(
            name="Random modifier",
            passed=False,
            detail="Policy learner failed with a random feature added — check data quality.",
        )

    gain = fit.value - original_value
    passed = gain <= original_se
    if passed:
        detail = f"value shifted by {gain:+.4f}  (≤ 1 SE = {original_se:.4f})  Noise adds no value, as expected."
    else:
        detail = (
            f"value shifted by {gain:+.4f}  (> 1 SE = {original_se:.4f})  "
            f"Adding a pure-noise feature improved the policy's value — "
            f"the honest value estimate is not reliable here."
        )
    return RefutationCheck(name="Random modifier", passed=passed, detail=detail)
