from __future__ import annotations

import numpy as np
import pandas as pd

from ..estimators.dml import _fit_dml
from ._check import RefutationCheck, RefutationReport, _add_random_column, _shift_check

_PLACEBO_SEED = 99999


def _check_random_common_cause(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: set[str],
    ml_outcome,
    ml_treatment,
    n_folds: int,
    original_effect: float,
    original_se: float,
) -> RefutationCheck:
    """
    Add a random noise column as an extra confounder and re-fit.

    The nuisance models should learn to ignore pure noise, so the estimate
    should not move by more than one standard error.
    """
    augmented, col = _add_random_column(data)
    model = _fit_dml(augmented, treatment, outcome, adjustment_set | {col}, ml_outcome, ml_treatment, n_folds)
    return _shift_check(
        float(model.coef[0]), original_effect, original_se, "Adding a random common cause destabilised the estimate."
    )


def _check_placebo_treatment(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: set[str],
    ml_outcome,
    ml_treatment,
    n_folds: int,
) -> RefutationCheck:
    """
    Permute the treatment at random and re-fit.

    A permuted treatment has no effect by construction, so the placebo's own
    99% CI should cover zero (95% gave ~8% false alarms in simulation). The
    placebo is judged by its own SE rather than the original's: permuting
    leaves the real effect unexplained, so the placebo fit is noisier and the
    original SE would fail it far too often (~30% in simulation).
    """
    rng = np.random.default_rng(_PLACEBO_SEED)
    augmented = data.assign(**{treatment: rng.permutation(data[treatment].values)})
    model = _fit_dml(augmented, treatment, outcome, adjustment_set, ml_outcome, ml_treatment, n_folds)
    lo, hi = (float(v) for v in model.confint(level=0.99).iloc[0])
    passed = lo <= 0 <= hi
    detail = f"placebo effect = {float(model.coef[0]):.4f}  (99% CI [{lo:.4f}, {hi:.4f}])"
    if passed:
        detail += "  Permuting the treatment yields no detectable effect, as expected."
    else:
        detail += (
            "  A randomly permuted treatment produced a significant effect — the nuisance models may be overfitting."
        )
    return RefutationCheck(name="Placebo treatment", passed=passed, detail=detail)


_OVERLAP_BOUNDS = (0.05, 0.95)
_OVERLAP_MAX_SHARE = 0.10
_TREATMENT_R2_MAX = 0.95


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def _check_overlap(model) -> RefutationCheck:
    """
    Share of units whose cross-fitted propensity score is outside [0.05, 0.95].

    Inverse-propensity weights explode near 0 and 1, so if many units are
    (almost) certain to be treated or untreated the estimate rests on a few
    heavily weighted comparisons — an overlap problem no learner can fix.
    """
    lo, hi = _OVERLAP_BOUNDS
    ps = model.predictions["ml_m"].ravel()
    share = float(np.mean((ps < lo) | (ps > hi)))
    passed = share <= _OVERLAP_MAX_SHARE
    detail = f"{share:.1%} of units have propensity outside [{lo}, {hi}]  (limit {_OVERLAP_MAX_SHARE:.0%})"
    if not passed:
        detail += (
            "  Some units are almost certain to be treated or untreated; the estimate "
            "extrapolates to them. Consider restricting to the region of common support."
        )
    return RefutationCheck(name="Overlap", passed=passed, detail=detail)


def _check_nuisance_fit(model, data: pd.DataFrame, treatment: str, outcome: str) -> RefutationCheck:
    """
    Out-of-fold R² of both nuisance models.

    The outcome model must beat predicting the mean (otherwise it is adding
    noise, usually from overfitting). The treatment model must not be close
    to perfect — if confounders almost determine the treatment, there is
    hardly any independent variation left to estimate the effect from.
    """
    y = data[outcome].to_numpy(dtype=float)
    d = data[treatment].to_numpy(dtype=float)
    preds = {k: v.ravel() for k, v in model.predictions.items()}
    y_hat = preds["ml_l"] if "ml_l" in preds else d * preds["ml_g1"] + (1 - d) * preds["ml_g0"]
    r2_y, r2_d = _r2(y, y_hat), _r2(d, preds["ml_m"])

    problems = []
    if r2_y < 0:
        problems.append("Outcome model predicts worse than the mean — likely overfitting; try a simpler learner.")
    if r2_d > _TREATMENT_R2_MAX:
        problems.append("Confounders almost fully determine treatment — little variation left to estimate from.")
    detail = f"out-of-fold R²: outcome = {r2_y:.3f}, treatment = {r2_d:.3f}"
    return RefutationCheck(name="Nuisance model fit", passed=not problems, detail="  ".join([detail, *problems]))


class DoubleMLRefutationReport(RefutationReport):
    """
    Results of refutation checks run against a Double ML estimation.

    Obtain via ``DoubleMLResult.refute(data)``.

    Example::

        result = DoubleML(dag, treatment="education", outcome="income").fit(df)
        report = result.refute(df)
        print(report.summary())
    """

    def _header_lines(self) -> list[str]:
        return [f"Double ML Refutation Report: {self._treatment} → {self._outcome}"]
