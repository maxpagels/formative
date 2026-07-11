from __future__ import annotations

import numpy as np
import pandas as pd

_RCC_SEED = 54321


def _add_random_column(data: pd.DataFrame, seed: int = _RCC_SEED) -> tuple[pd.DataFrame, str]:
    """Return a copy of *data* with a random normal column added, plus the column name."""
    rng = np.random.default_rng(seed)
    col = "_rcc"
    while col in data.columns:
        col = "_" + col
    return data.assign(**{col: rng.normal(size=len(data))}), col


def _shift_check(new_effect: float, original_effect: float, original_se: float, fail_suffix: str) -> RefutationCheck:
    """Random-common-cause verdict: pass iff the refitted estimate moved by at most one standard error."""
    shift = abs(new_effect - original_effect)
    passed = shift <= original_se
    if passed:
        detail = f"estimate shifted by {shift:.4f}  (≤ 1 SE = {original_se:.4f})"
    else:
        detail = f"estimate shifted by {shift:.4f}  (> 1 SE = {original_se:.4f})  {fail_suffix}"
    return RefutationCheck(name="Random common cause", passed=passed, detail=detail)


def _placebo_check(
    name: str,
    label: str,
    placebo_effect: float,
    original_se: float,
    pass_suffix: str,
    fail_suffix: str,
) -> RefutationCheck:
    """Placebo verdict: pass iff the placebo estimate is within one standard error of zero."""
    passed = abs(placebo_effect) <= original_se
    if passed:
        detail = f"{label} = {placebo_effect:.4f}  (≤ 1 SE = {original_se:.4f})  {pass_suffix}"
    else:
        detail = f"{label} = {placebo_effect:.4f}  (> 1 SE = {original_se:.4f})  {fail_suffix}"
    return RefutationCheck(name=name, passed=passed, detail=detail)


class RefutationCheck:
    """Result of a single refutation check."""

    def __init__(self, name: str, passed: bool, detail: str) -> None:
        self.name = name
        self.passed = passed
        self.detail = detail

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"RefutationCheck({status!r}, {self.name!r})"


class RefutationReport:
    """
    Base class for refutation reports.

    Subclasses implement ``_header_lines()`` to supply the method-specific
    title shown at the top of ``summary()``. All common logic — ``checks``,
    ``passed``, ``failed_checks``, ``summary()``, and ``__repr__`` — lives here.
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

    def _header_lines(self) -> list[str]:
        raise NotImplementedError

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
        """Formatted report showing each check result and the overall verdict."""
        lines = ["", *self._header_lines(), "─" * 50]
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
