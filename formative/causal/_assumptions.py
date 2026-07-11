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
