"""
Interaction-based heterogeneous treatment effects (CATE).

Shared machinery for estimators that accept an ``effect_modifier``: the model
``outcome ~ treatment * C(modifier) + controls`` is fitted once, and per-group
effects, a homogeneity test, and a sample-share-weighted average effect are
extracted from it. Used by ``OLSObservational`` and ``RCT``; the refutation
checks in ``formative/causal/refutations/cate.py`` refit via ``_fit_cate``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..dag import DAG
from ..refutations._check import Assumption

CATE_ASSUMPTIONS: list[Assumption] = [
    Assumption("Correct specification of effect modification (linear interaction form)", testable=False),
    Assumption("Effect modifier is measured pre-treatment", testable=False),
]


@dataclass(frozen=True)
class GroupEffect:
    """The estimated treatment effect within one level of the effect modifier."""

    level: object
    """The modifier level this effect applies to."""

    effect: float
    """Point estimate of the treatment effect for this group."""

    std_err: float
    """Standard error of the group effect estimate."""

    conf_int: tuple[float, float]
    """95% confidence interval for the group effect."""

    pvalue: float
    """p-value for the group effect (``H0: effect = 0``)."""

    n: int
    """Number of observations in this group."""


@dataclass(frozen=True)
class _CATEFit:
    """Everything extracted from one fit of the interaction model."""

    result: object  # statsmodels result for outcome ~ treatment * C(modifier) + controls
    group_effects: list[GroupEffect]
    homogeneity_fstat: float
    homogeneity_pvalue: float
    effect: float  # sample-share-weighted average of the group effects
    std_err: float
    conf_int: tuple[float, float]
    pvalue: float


def _validate_modifier_dag(dag: DAG, treatment: str, outcome: str, modifier: str) -> None:
    """Structural checks on the effect modifier. Raises ``ValueError`` on violation."""
    if modifier in (treatment, outcome):
        raise ValueError("Effect modifier must be different from treatment and outcome.")
    if modifier not in dag.nodes:
        raise ValueError(f"Effect modifier '{modifier}' is not a node in the DAG. Known nodes: {sorted(dag.nodes)}")
    if modifier in dag.descendants(treatment):
        raise ValueError(
            f"Effect modifier '{modifier}' is a descendant of treatment '{treatment}' in the DAG. "
            f"Conditioning on a post-treatment variable (a mediator) produces biased, artificial "
            f"heterogeneity. Use a modifier that is measured before treatment."
        )
    if outcome not in dag.descendants(modifier):
        raise ValueError(
            f"Effect modifier '{modifier}' is not an ancestor of outcome '{outcome}' in the DAG. "
            f"A variable that modifies the effect on '{outcome}' is a cause of it — assert the "
            f"corresponding path, e.g. dag.assume('{modifier}').causes('{outcome}')."
        )


def _validate_modifier_data(data: pd.DataFrame, treatment: str, modifier: str) -> None:
    """Data checks on the effect modifier column. Raises ``ValueError`` on violation."""
    if modifier not in data.columns:
        raise ValueError(f"Effect modifier column '{modifier}' not found in dataframe.")
    levels = data[modifier].dropna().unique()
    if len(levels) < 2:
        raise ValueError(f"Effect modifier '{modifier}' must have at least 2 levels. Found: {list(levels)}")
    for lvl in levels:
        if data.loc[data[modifier] == lvl, treatment].nunique() < 2:
            raise ValueError(
                f"Treatment '{treatment}' does not vary within {modifier} = {lvl!r}, "
                f"so the effect for that group is not estimable."
            )


def _fit_cate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifier: str,
    controls: set[str],
) -> _CATEFit:
    """
    Fit ``outcome ~ treatment * C(modifier) + controls`` and extract per-group effects.

    The effect for modifier level *g* is the linear combination
    ``β_treatment + β_treatment:C(modifier)[T.g]`` (zero interaction term for the
    reference level); its uncertainty comes from the full parameter covariance
    matrix via ``t_test``. The headline effect is the sample-share-weighted
    average of the group effects — with interactions, the raw treatment
    coefficient is only the reference group's effect. The homogeneity F-test
    checks that all interaction coefficients are jointly zero.
    """
    control_cols = sorted(set(controls) - {modifier})
    rhs = f"{treatment} * C({modifier})"
    if control_cols:
        rhs += " + " + " + ".join(control_cols)
    result = smf.ols(f"{outcome} ~ {rhs}", data=data).fit()

    params = list(result.params.index)
    t_idx = params.index(treatment)
    prefix = f"{treatment}:C({modifier})[T."
    interaction_idx = {p[len(prefix) : -1]: i for i, p in enumerate(params) if p.startswith(prefix)}

    counts = data[modifier].value_counts()
    try:
        levels = sorted(counts.index)
    except TypeError:
        levels = sorted(counts.index, key=str)

    unmatched = [lvl for lvl in levels if str(lvl) not in interaction_idx]
    if len(unmatched) != 1:
        raise RuntimeError(
            f"Could not align levels of '{modifier}' with model parameters "
            f"(levels without an interaction term: {unmatched}; expected exactly one reference level)."
        )

    def _t_test(contrast: np.ndarray) -> tuple[float, float, tuple[float, float], float]:
        tt = result.t_test(contrast)
        lo, hi = (float(x) for x in np.asarray(tt.conf_int()).ravel())
        return (
            float(np.asarray(tt.effect).ravel()[0]),
            float(np.asarray(tt.sd).ravel()[0]),
            (lo, hi),
            float(np.asarray(tt.pvalue).ravel()[0]),
        )

    n_total = int(counts.sum())
    group_effects = []
    weighted_contrast = np.zeros(len(params))
    for lvl in levels:
        contrast = np.zeros(len(params))
        contrast[t_idx] = 1.0
        i = interaction_idx.get(str(lvl))
        if i is not None:
            contrast[i] = 1.0
        effect, sd, ci, p = _t_test(contrast)
        n_g = int(counts[lvl])
        group_effects.append(GroupEffect(level=lvl, effect=effect, std_err=sd, conf_int=ci, pvalue=p, n=n_g))
        weighted_contrast += (n_g / n_total) * contrast

    effect, sd, ci, p = _t_test(weighted_contrast)

    restriction = np.zeros((len(interaction_idx), len(params)))
    for row, i in enumerate(interaction_idx.values()):
        restriction[row, i] = 1.0
    f_test = result.f_test(restriction)

    return _CATEFit(
        result=result,
        group_effects=group_effects,
        homogeneity_fstat=float(np.asarray(f_test.fvalue).ravel()[0]),
        homogeneity_pvalue=float(np.asarray(f_test.pvalue).ravel()[0]),
        effect=effect,
        std_err=sd,
        conf_int=ci,
        pvalue=p,
    )


class _CATEResultMixin:
    """
    Adds per-group effects to a result class.

    Subclasses set ``self._cate`` (a ``_CATEFit``) and ``self._modifier`` in
    ``__init__``. Must precede the base result class in the MRO so that the
    weighted-average ``effect``/``std_err``/``conf_int``/``pvalue`` override
    the ones read from the raw treatment coefficient, which under interaction
    coding is only the reference group's effect.
    """

    _cate: _CATEFit
    _modifier: str

    @property
    def effect(self) -> float:
        """Sample-share-weighted average of the per-group treatment effects."""
        return self._cate.effect

    @property
    def std_err(self) -> float:
        """Standard error of the weighted average effect."""
        return self._cate.std_err

    @property
    def conf_int(self) -> tuple[float, float]:
        """95% confidence interval for the weighted average effect."""
        return self._cate.conf_int

    @property
    def pvalue(self) -> float:
        """p-value for the weighted average effect (``H0: effect = 0``)."""
        return self._cate.pvalue

    @property
    def effect_modifier(self) -> str:
        """Name of the effect modifier column."""
        return self._modifier

    @property
    def effect_by_group(self) -> list[GroupEffect]:
        """Treatment effect within each level of the effect modifier, sorted by level."""
        return list(self._cate.group_effects)

    @property
    def homogeneity_fstat(self) -> float:
        """F-statistic testing that all treatment × modifier interactions are jointly zero."""
        return self._cate.homogeneity_fstat

    @property
    def homogeneity_pvalue(self) -> float:
        """p-value of the homogeneity test. Small values indicate genuine effect heterogeneity."""
        return self._cate.homogeneity_pvalue

    def decide_by_group(self, cost: float, benefit: float) -> dict:
        """
        Compute a cost-benefit decision analysis per modifier level.

        Parameters
        ----------
        cost : float
            Cost per unit of treatment applied.
        benefit : float
            Benefit (revenue, utility, etc.) per unit increase in the outcome.

        Returns
        -------
        dict
            Mapping of modifier level → ``DecisionReport``, so different
            groups can receive different treatment decisions.
        """
        from ..decision import _decide

        return {
            g.level: _decide(
                g.effect,
                g.std_err,
                g.conf_int,
                f"{self._treatment} | {self._modifier} = {g.level}",
                self._outcome,
                cost,
                benefit,
            )
            for g in self._cate.group_effects
        }

    def _extra_summary_lines(self) -> list[str]:
        width = max(len(str(g.level)) for g in self._cate.group_effects)
        lines = [
            "",
            f"  Effect by {self._modifier}  "
            f"(homogeneity: F = {self._cate.homogeneity_fstat:.2f}, p = {self._cate.homogeneity_pvalue:.4f})",
            "  " + "┄" * 48,
        ]
        for g in self._cate.group_effects:
            lo, hi = g.conf_int
            lines.append(f"    {str(g.level):<{width}} : {g.effect:>10.4f}  [{lo:.4f}, {hi:.4f}]  n={g.n}")
        return lines
