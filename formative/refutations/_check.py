from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assumption:
    """A single modelling assumption, with a testability flag."""

    name: str
    testable: bool


class RefutationCheck:
    """Result of a single refutation check."""

    def __init__(self, name: str, passed: bool, detail: str) -> None:
        self.name = name
        self.passed = passed
        self.detail = detail

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"RefutationCheck({status!r}, {self.name!r})"
