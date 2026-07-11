"""
Doubly robust policy learning (Athey & Wager) on RCT results.

``RCTResult.learn_policy()`` turns an experiment into a treatment *rule*: it
computes cross-fitted AIPW scores for every unit, converts them to per-unit
net benefit via the user's cost/benefit, and searches exhaustively for the
shallow decision tree over discrete features that maximises total net
benefit. The reported policy value is estimated honestly — each unit is
scored by a tree learned without its fold — and expressed as the advantage
over the best constant policy (treat everyone or no one), so a policy that
learned nothing reads as ≈ 0.

Refutation checks in ``formative/causal/refutations/policy.py`` re-run
``_fit_policy`` on perturbed data; DAG validation happens once in
``_learn_policy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._assumptions import Assumption
from ._cate import _validate_modifier_dag

_FOLD_SEED = 13579
_N_FOLDS = 5
_MIN_UNITS_PER_ARM = 2 * _N_FOLDS
_MAX_LEVELS = 20
_TIE_EPS = 1e-9  # relative to Σ|γ|; see _learn_tree

POLICY_ASSUMPTIONS: list[Assumption] = [
    Assumption("Random assignment of treatment", testable=False),
    Assumption("Stable Unit Treatment Value Assumption (SUTVA)", testable=False),
    Assumption("Policy features are measured pre-treatment", testable=False),
    Assumption("Cost and benefit are constant across units", testable=False),
]


@dataclass(frozen=True)
class PolicyNode:
    """
    One internal split of a learned policy tree.

    Units where ``feature == level`` follow ``if_true``, all others follow
    ``if_false``. A child is either another ``PolicyNode`` or a leaf action:
    ``True`` (treat) / ``False`` (don't treat).
    """

    feature: str
    """Column the split tests."""

    level: object
    """The level compared against: the split is ``feature == level``."""

    if_true: "PolicyNode | bool"
    """Subtree or leaf action for units where the condition holds."""

    if_false: "PolicyNode | bool"
    """Subtree or leaf action for the remaining units."""


def _assign_tree(tree: PolicyNode | bool, data: pd.DataFrame) -> np.ndarray:
    """Evaluate a policy tree on *data*, returning a boolean treat/don't array."""
    if isinstance(tree, bool):
        return np.full(len(data), tree, dtype=bool)
    cond = data[tree.feature].to_numpy() == tree.level
    out = np.empty(len(data), dtype=bool)
    out[cond] = _assign_tree(tree.if_true, data.loc[cond])
    out[~cond] = _assign_tree(tree.if_false, data.loc[~cond])
    return out


def _tree_features(tree: PolicyNode | bool) -> set[str]:
    """All feature names the tree splits on."""
    if isinstance(tree, bool):
        return set()
    return {tree.feature} | _tree_features(tree.if_true) | _tree_features(tree.if_false)


def _rules_lines(tree: PolicyNode | bool) -> list[str]:
    """Render the tree as one readable line per treated path."""
    if isinstance(tree, bool):
        return ["treat everyone" if tree else "treat no one"]

    paths: list[list[str]] = []

    def walk(node: PolicyNode | bool, conds: list[str]) -> None:
        if isinstance(node, bool):
            if node:
                paths.append(conds)
            return
        walk(node.if_true, conds + [f"{node.feature} = {node.level}"])
        walk(node.if_false, conds + [f"{node.feature} ≠ {node.level}"])

    walk(tree, [])
    if not paths:
        return ["treat no one"]
    return [f"treat if {' and '.join(conds)}" for conds in paths] + ["otherwise don't treat"]


def _simplify_tree(tree: PolicyNode | bool, data: pd.DataFrame) -> PolicyNode | bool:
    """
    Drop splits that don't change the assignment on *data*.

    The exhaustive search keeps the first of several equal-value trees, which
    can carry redundant conditions (e.g. a root split whose subtree already
    implies it). Replacing a node by one of its children is safe whenever the
    assignment is identical over the sample — same assignment, same value.
    """
    if isinstance(tree, bool):
        return tree
    cond = data[tree.feature].to_numpy() == tree.level
    left = _simplify_tree(tree.if_true, data.loc[cond])
    right = _simplify_tree(tree.if_false, data.loc[~cond])
    if isinstance(left, bool) and isinstance(right, bool) and left == right:
        return left
    tree = PolicyNode(tree.feature, tree.level, left, right)
    full = _assign_tree(tree, data)
    for child in (left, right):
        if np.array_equal(_assign_tree(child, data), full):
            return child
    return tree


def _dr_scores(
    W: np.ndarray,
    Y: np.ndarray,
    X: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    """
    Cross-fitted AIPW scores for the treatment effect of each unit.

    For each fold, outcome models ``μ̂₁``/``μ̂₀`` are least-squares fits of the
    outcome on the feature dummies over the *other* folds' treated/control
    units, and the propensity is the other folds' treated share (constant —
    this is what restricts v1 to RCTs). The held-out score is
    ``μ̂₁ − μ̂₀ + W/ê·(Y − μ̂₁) − (1−W)/(1−ê)·(Y − μ̂₀)``.
    """
    gamma = np.empty(len(W))
    for k in np.unique(folds):
        test = folds == k
        train = ~test
        e_hat = W[train].mean()
        beta1, *_ = np.linalg.lstsq(X[train & (W == 1)], Y[train & (W == 1)], rcond=None)
        beta0, *_ = np.linalg.lstsq(X[train & (W == 0)], Y[train & (W == 0)], rcond=None)
        mu1 = X[test] @ beta1
        mu0 = X[test] @ beta0
        gamma[test] = mu1 - mu0 + W[test] / e_hat * (Y[test] - mu1) - (1 - W[test]) / (1 - e_hat) * (Y[test] - mu0)
    return gamma


def _learn_tree(
    gamma: np.ndarray,
    M: np.ndarray,
    candidates: list[tuple[str, object]],
    max_depth: int,
) -> tuple[PolicyNode | bool, float]:
    """
    Exhaustive search for the depth ≤ ``max_depth`` tree maximising total net benefit.

    ``M`` is the (candidate × unit) boolean split matrix aligned with
    ``gamma``. A leaf treats iff its total score is positive; ties prefer
    shallower trees (a split must *strictly* beat not splitting). The tie
    tolerance scales with the score magnitudes so that float noise from
    summing the same scores in a different order can never look like an
    improvement.
    """
    total = float(gamma.sum())
    best_val = max(total, 0.0)
    best_tree: PolicyNode | bool = total > 0
    if max_depth == 0 or len(gamma) == 0:
        return best_tree, best_val

    tie_eps = _TIE_EPS * (1.0 + float(np.abs(gamma).sum()))
    s1 = M @ gamma
    s0 = total - s1

    if max_depth == 1:
        vals = np.maximum(s1, 0.0) + np.maximum(s0, 0.0)
        i = int(np.argmax(vals))
        if vals[i] > best_val + tie_eps:
            feature, level = candidates[i]
            best_tree = PolicyNode(feature, level, bool(s1[i] > 0), bool(s0[i] > 0))
            best_val = float(vals[i])
        return best_tree, best_val

    for i, (feature, level) in enumerate(candidates):
        mask = M[i].astype(bool)
        left_tree, left_val = _learn_tree(gamma[mask], M[:, mask], candidates, max_depth - 1)
        right_tree, right_val = _learn_tree(gamma[~mask], M[:, ~mask], candidates, max_depth - 1)
        if left_val + right_val > best_val + tie_eps:
            best_val = left_val + right_val
            best_tree = PolicyNode(feature, level, left_tree, right_tree)
    return best_tree, best_val


@dataclass(frozen=True)
class _PolicyFit:
    """Everything computed by one run of the policy learner."""

    tree: PolicyNode | bool
    value: float  # honest advantage over the best constant policy, per unit
    value_se: float
    value_ci: tuple[float, float]
    coverage: float


def _fit_policy(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifiers: list[str],
    cost: float,
    benefit: float,
    max_depth: int,
) -> _PolicyFit:
    """
    Compute DR scores, the honest value estimate, and the final tree.

    The value is estimated out-of-fold: for each fold, a tree and the best
    constant policy are learned on the other folds, and held-out units
    contribute ``(π_tree − π_const)·γ``. The reported tree is re-learned on
    all units, so the value estimate describes the *procedure* — standard
    practice, since scoring the chosen tree in-sample would be optimistic.
    """
    W = data[treatment].to_numpy(dtype=float)
    Y = data[outcome].to_numpy(dtype=float)
    n = len(data)

    dummies = pd.get_dummies(data[modifiers].astype("category"), drop_first=True)
    X = np.column_stack([np.ones(n), dummies.to_numpy(dtype=float)])

    # Stratified fold assignment: spread each arm evenly so every training
    # split contains both treated and control units.
    rng = np.random.default_rng(_FOLD_SEED)
    folds = np.empty(n, dtype=int)
    for arm in (0, 1):
        idx = rng.permutation(np.flatnonzero(W == arm))
        folds[idx] = np.arange(len(idx)) % _N_FOLDS

    gamma = benefit * _dr_scores(W, Y, X, folds) - cost

    candidates = [(m, lvl) for m in modifiers for lvl in pd.unique(data[m].dropna())]
    M = np.stack([data[m].to_numpy() == lvl for m, lvl in candidates]).astype(float)

    d = np.empty(n)
    for k in range(_N_FOLDS):
        test = folds == k
        train = ~test
        fold_tree, _ = _learn_tree(gamma[train], M[:, train], candidates, max_depth)
        const = float(gamma[train].sum() > 0)
        d[test] = (_assign_tree(fold_tree, data.loc[test]).astype(float) - const) * gamma[test]

    value = float(d.mean())
    value_se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    value_ci = (value - 1.96 * value_se, value + 1.96 * value_se)

    tree, _ = _learn_tree(gamma, M, candidates, max_depth)
    tree = _simplify_tree(tree, data)
    coverage = float(_assign_tree(tree, data).mean())

    return _PolicyFit(tree=tree, value=value, value_se=value_se, value_ci=value_ci, coverage=coverage)


class PolicyResult:
    """
    A treatment assignment policy learned from an RCT.

    Produced by ``RCTResult.learn_policy()``. The policy is a shallow decision
    tree over discrete pre-treatment features, chosen to maximise total net
    benefit (``benefit × effect − cost`` per treated unit) under doubly robust
    scoring. ``value`` is an honest (cross-fitted) estimate of how much net
    benefit per unit the policy adds over the best constant policy — treating
    everyone or no one, whichever is better — so ``value ≈ 0`` means the
    learned targeting is worthless even if treating is not.
    """

    _ASSUMPTIONS = POLICY_ASSUMPTIONS

    def __init__(
        self,
        fit: _PolicyFit,
        treatment: str,
        outcome: str,
        modifiers: list[str],
        cost: float,
        benefit: float,
        max_depth: int,
        dag,
        n: int,
    ) -> None:
        self._fit = fit
        self._treatment = treatment
        self._outcome = outcome
        self._modifiers = modifiers
        self._cost = cost
        self._benefit = benefit
        self._max_depth = max_depth
        self._dag = dag
        self._n = n

    @property
    def tree(self) -> PolicyNode | bool:
        """The learned policy tree (``True``/``False`` if a constant policy won)."""
        return self._fit.tree

    @property
    def rules(self) -> str:
        """The policy as readable text, one line per treated path."""
        return "\n".join(_rules_lines(self._fit.tree))

    @property
    def value(self) -> float:
        """Honest per-unit net-benefit advantage over the best constant policy."""
        return self._fit.value

    @property
    def value_se(self) -> float:
        """Standard error of the policy value estimate."""
        return self._fit.value_se

    @property
    def value_ci(self) -> tuple[float, float]:
        """95% confidence interval for the policy value."""
        return self._fit.value_ci

    @property
    def coverage(self) -> float:
        """Share of the sample the policy treats."""
        return self._fit.coverage

    @property
    def modifiers(self) -> list[str]:
        """Candidate feature columns the tree was allowed to split on."""
        return list(self._modifiers)

    @property
    def assumptions(self) -> list[Assumption]:
        """Modelling assumptions required for the policy value to be causal."""
        return list(self._ASSUMPTIONS)

    def assign(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply the learned policy to (new) data.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain the modifier columns the tree splits on.

        Returns
        -------
        pd.Series
            Boolean treat/don't-treat assignment, aligned with ``data``'s index.
        """
        return pd.Series(_assign_tree(self._fit.tree, data), index=data.index, name="treat")

    def summary(self) -> str:
        """Concise tabular summary of the learned rule, its value, and assumptions."""
        lo, hi = self.value_ci
        lines = [
            "",
            f"Learned Policy: {self._treatment} → {self._outcome}",
            "  Estimand: policy value vs best constant policy (doubly robust)",
            "─" * 50,
            "  Policy rule",
            "  " + "┄" * 48,
        ]
        lines += [f"    {line}" for line in _rules_lines(self._fit.tree)]
        lines += [
            "",
            f"  Value vs constant    : {self.value:>+10.4f} per unit",
            f"  Std. error           : {self.value_se:>10.4f}",
            f"  95% CI               : [{lo:+.4f}, {hi:+.4f}]",
            f"  Coverage             : {self.coverage:>10.1%}",
            f"  Cost / benefit       : {self._cost:.4g} / {self._benefit:.4g}",
            f"  N                    : {self._n:>10}",
        ]
        lines += self._assumptions_lines()
        return "\n".join(lines)

    def _assumptions_lines(self, width: int = 48) -> list[str]:
        lines = ["", "  Assumptions", "  " + "┄" * width]
        lines += [f"  {a.fmt_tag()}  {a.name}" for a in self._ASSUMPTIONS]
        lines.append("")
        return lines

    def executive_summary(self) -> str:
        """Narrative explanation of the method, learned rule, and caveats."""
        from .._explain import explain_policy

        return explain_policy(self)

    def refute(self, data: pd.DataFrame):
        """
        Run refutation checks against this learned policy.

        Currently runs:

        - **Placebo modifiers**: permutes every candidate feature column and
          re-learns. With the features scrambled there is nothing to target,
          so the placebo policy's value should not be significantly positive.
        - **Random modifier**: adds a pure-noise feature to the candidates and
          re-learns. The honest value should not improve by more than one
          standard error — if noise helps, the value estimate is not trustworthy.

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``learn_policy()``.
        """
        from ..refutations.policy import (
            PolicyRefutationReport,
            _check_placebo_modifiers,
            _check_random_modifier,
        )

        args = (data, self._treatment, self._outcome, self._modifiers, self._cost, self._benefit, self._max_depth)
        checks = [
            _check_placebo_modifiers(*args),
            _check_random_modifier(*args, self.value, self.value_se),
        ]
        return PolicyRefutationReport(checks=checks, treatment=self._treatment, outcome=self._outcome)

    def __repr__(self) -> str:
        return self.summary()


def _validate_policy_inputs(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifiers: list[str],
    max_depth: int,
    dag,
) -> None:
    if not modifiers:
        raise ValueError("modifiers must be a non-empty list of candidate feature columns.")
    if len(set(modifiers)) != len(modifiers):
        raise ValueError("modifiers must not contain duplicates.")
    if max_depth not in (1, 2):
        raise ValueError(f"max_depth must be 1 or 2 (got {max_depth}) — deeper trees stop being auditable.")

    for m in modifiers:
        _validate_modifier_dag(dag, treatment, outcome, m)
        if m not in data.columns:
            raise ValueError(f"Modifier column '{m}' not found in dataframe.")
        n_levels = data[m].nunique()
        if n_levels < 2:
            raise ValueError(f"Modifier '{m}' must have at least 2 levels. Found: {list(data[m].dropna().unique())}")
        if n_levels > _MAX_LEVELS:
            raise ValueError(
                f"Modifier '{m}' has {n_levels} levels (max {_MAX_LEVELS}). "
                f"Policy trees need discrete features — bin continuous or high-cardinality columns first."
            )

    used = [treatment, outcome, *modifiers]
    with_nan = [c for c in used if data[c].isna().any()]
    if with_nan:
        raise ValueError(f"Columns {with_nan} contain missing values — drop or impute them before learning a policy.")

    w = set(data[treatment].unique())
    if w != {0, 1}:
        raise ValueError(f"Policy learning requires a binary 0/1 treatment. '{treatment}' has values: {sorted(w)}")
    counts = data[treatment].value_counts()
    if counts.min() < _MIN_UNITS_PER_ARM:
        raise ValueError(
            f"Need at least {_MIN_UNITS_PER_ARM} units in each treatment arm for "
            f"{_N_FOLDS}-fold cross-fitting; smallest arm has {counts.min()}."
        )


def _learn_policy(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifiers: list[str],
    cost: float,
    benefit: float,
    max_depth: int,
    dag,
) -> PolicyResult:
    """Validate inputs, run the policy learner, and wrap the result."""
    modifiers = list(modifiers)
    _validate_policy_inputs(data, treatment, outcome, modifiers, max_depth, dag)
    fit = _fit_policy(data, treatment, outcome, modifiers, cost, benefit, max_depth)
    return PolicyResult(fit, treatment, outcome, modifiers, cost, benefit, max_depth, dag, len(data))
