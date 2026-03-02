from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assumption:
    """
    A single modelling assumption required for causal identification.

    Every estimator exposes its assumptions via ``result.assumptions``,
    a list of ``Assumption`` objects. Each assumption has a human-readable
    name and a ``testable`` flag indicating whether it can be empirically
    checked in the data or must be justified on substantive grounds.
    """

    name: str
    """Human-readable description of the assumption."""

    testable: bool
    """``True`` if the assumption can be empirically checked; ``False`` if it rests on domain knowledge."""

    def fmt_tag(self) -> str:
        """Return a fixed-width bracketed testability label for use in summary output."""
        return "[  testable  ]" if self.testable else "[ untestable ]"


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
