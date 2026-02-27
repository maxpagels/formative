from __future__ import annotations
from ._exceptions import GraphError


class DAG:
    """
    A directed acyclic graph representing causal assumptions.

    Nodes are variable names (strings). Edges represent direct causal
    relationships: add_edge("A", "B") means "A causes B".

    Include latent variables (unobserved confounders) as nodes — just don't
    include their column in your dataframe. The estimator will detect that
    they are unobserved when you call fit().

    Usage
    -----
        dag = DAG()
        dag.causes("ability", "education")   # latent — not in dataframe
        dag.causes("ability", "income")
        dag.causes("education", "income")
    """

    def __init__(self) -> None:
        self._edges: list[tuple[str, str]] = []

    # ── Building the graph ────────────────────────────────────────────────────

    def causes(self, cause: str, effect: str) -> DAG:
        """
        Assert that cause → effect. Returns self for chaining.

            dag.causes("ability", "education")  # I assume ability causes education
        """
        if cause == effect:
            raise GraphError(f"Self-loops are not allowed: '{cause}'")
        if (cause, effect) in self._edges:
            raise GraphError(f"'{cause}' → '{effect}' already asserted")
        self._edges.append((cause, effect))
        if self._has_cycle():
            self._edges.pop()
            raise GraphError(
                f"Asserting '{cause}' → '{effect}' would create a cycle. "
                f"Causal graphs must be acyclic (DAGs)."
            )
        return self

    # ── Graph properties ──────────────────────────────────────────────────────

    @property
    def nodes(self) -> set[str]:
        """All nodes in the graph."""
        result: set[str] = set()
        for cause, effect in self._edges:
            result.add(cause)
            result.add(effect)
        return result

    @property
    def edges(self) -> list[tuple[str, str]]:
        """All directed edges as (cause, effect) pairs."""
        return list(self._edges)

    def parents(self, node: str) -> set[str]:
        """Direct causes of node (nodes with an edge into node)."""
        return {cause for cause, effect in self._edges if effect == node}

    def children(self, node: str) -> set[str]:
        """Direct effects of node (nodes with an edge out of node)."""
        return {effect for cause, effect in self._edges if cause == node}

    def ancestors(self, node: str) -> set[str]:
        """All nodes with a directed path leading to node."""
        result: set[str] = set()
        queue = list(self.parents(node))
        while queue:
            current = queue.pop()
            if current not in result:
                result.add(current)
                queue.extend(self.parents(current))
        return result

    def descendants(self, node: str) -> set[str]:
        """All nodes reachable from node via directed paths."""
        result: set[str] = set()
        queue = list(self.children(node))
        while queue:
            current = queue.pop()
            if current not in result:
                result.add(current)
                queue.extend(self.children(current))
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _has_cycle(self) -> bool:
        """Kahn's algorithm: returns True if the current edge list contains a cycle."""
        in_degree: dict[str, int] = {n: 0 for n in self.nodes}
        for _, effect in self._edges:
            in_degree[effect] += 1

        queue = [n for n, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return visited != len(self.nodes)

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self._edges:
            return "DAG (empty)"
        lines = ["DAG:"]
        for cause, effect in self._edges:
            lines.append(f"  {cause} → {effect}")
        return "\n".join(lines)
