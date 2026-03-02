from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..dag import DAG
from .._exceptions import IdentificationError
from ..refutations._check import Assumption

MATCHING_ASSUMPTIONS: list[Assumption] = [
    Assumption("Conditional independence: no unobserved confounders given matched variables", testable=False),
    Assumption("Common support: overlap exists in characteristics between groups", testable=True),
    Assumption("Correct specification of the matching variables", testable=False),
    Assumption("Stable Unit Treatment Value Assumption (SUTVA)", testable=False),
]

_BOOTSTRAP_N    = 500
_BOOTSTRAP_SEED = 42


# ── Private helpers (also imported by formative/refutations/matching.py) ──────

def _propensity_scores(
    data: pd.DataFrame,
    treatment: str,
    adjustment_set: set[str],
) -> np.ndarray:
    """Fit logistic regression and return propensity scores as a 1-D array."""
    if adjustment_set:
        rhs = " + ".join(sorted(adjustment_set))
        ps = smf.logit(f"{treatment} ~ {rhs}", data=data).fit(disp=0).predict()
    else:
        # Intercept-only: all units get the same PS (treatment base rate).
        ps = smf.logit(f"{treatment} ~ 1", data=data).fit(disp=0).predict()
    return np.asarray(ps)


def _att_from_ps(
    treatment: np.ndarray,
    outcome: np.ndarray,
    ps: np.ndarray,
) -> float:
    """
    1-to-1 nearest-neighbour matching on propensity score (with replacement).
    Returns the ATT: mean outcome difference across matched pairs.
    """
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("Sample must contain both treated and control units.")

    ps_treated = ps[treated_idx]
    ps_control = ps[control_idx]

    # For each treated unit find the nearest control unit by PS distance.
    dists = np.abs(ps_treated[:, None] - ps_control[None, :])
    matched = control_idx[np.argmin(dists, axis=1)]

    return float(np.mean(outcome[treated_idx] - outcome[matched]))


# ── Result ─────────────────────────────────────────────────────────────────────

class MatchingResult:
    """
    The result of a propensity score matching estimation.

    Holds the ATT point estimate alongside its unadjusted counterpart (naive
    mean difference), so you can see the confounding bias that matching
    corrects. Standard errors and CIs are computed via bootstrap over the
    full matching procedure.
    """

    def __init__(
        self,
        att: float,
        unadjusted_effect: float,
        bootstrap_atts: np.ndarray,
        treatment: str,
        outcome: str,
        adjustment_set: set[str],
        dag,
    ) -> None:
        self._att = att
        self._unadjusted_effect = unadjusted_effect
        self._bootstrap_atts = bootstrap_atts
        self._treatment = treatment
        self._outcome = outcome
        self._adjustment_set = adjustment_set
        self._dag = dag

    @property
    def effect(self) -> float:
        """ATT: average treatment effect on the treated."""
        return self._att

    @property
    def unadjusted_effect(self) -> float:
        """Naive mean difference Y|T=1 minus Y|T=0, no matching."""
        return self._unadjusted_effect

    @property
    def std_err(self) -> float:
        """Bootstrap standard error of the ATT."""
        return float(np.std(self._bootstrap_atts, ddof=1))

    @property
    def conf_int(self) -> tuple[float, float]:
        """Bootstrap percentile 95% confidence interval."""
        return (
            float(np.percentile(self._bootstrap_atts, 2.5)),
            float(np.percentile(self._bootstrap_atts, 97.5)),
        )

    @property
    def pvalue(self) -> float:
        """Two-sided p-value for the ATT (``H0: ATT = 0``), via z-test."""
        import scipy.stats as _st
        z = abs(self._att) / self.std_err
        return float(2.0 * _st.norm.sf(z))

    @property
    def adjustment_set(self) -> set[str]:
        """Observed confounders used in the propensity score model."""
        return self._adjustment_set

    @property
    def bootstrap_atts(self) -> np.ndarray:
        """Full array of per-bootstrap ATT values, for diagnostics."""
        return self._bootstrap_atts.copy()

    @property
    def assumptions(self) -> list[Assumption]:
        """Modelling assumptions required for a causal interpretation."""
        return list(MATCHING_ASSUMPTIONS)

    def executive_summary(self) -> str:
        """Narrative explanation of the method, DAG, assumptions, and result."""
        from .._explain import explain_matching
        return explain_matching(self)

    def summary(self) -> str:
        lo, hi = self.conf_int
        adj = sorted(self._adjustment_set)
        bias = self.unadjusted_effect - self.effect

        lines = [
            "",
            f"PSM Causal Effect: {self._treatment} → {self._outcome}",
            f"  Estimand: ATT (average treatment effect on the treated)",
            "─" * 54,
        ]

        if adj:
            lines += [
                f"  ATT estimate         : {self.effect:>10.4f}  (controlling for: {', '.join(adj)})",
                f"  Unadjusted estimate  : {self.unadjusted_effect:>10.4f}  (naive mean difference)",
                f"  Confounding bias     : {bias:>+10.4f}",
            ]
        else:
            lines += [
                f"  ATT estimate         : {self.effect:>10.4f}  (no confounders in DAG)",
            ]

        lines += [
            "",
            f"  Std. error           : {self.std_err:>10.4f}  (bootstrap, N={_BOOTSTRAP_N})",
            f"  95% CI               : [{lo:.4f}, {hi:.4f}]  (bootstrap percentile)",
            f"  p-value              : {self.pvalue:>10.4f}",
            "",
            "  Matching: 1-to-1 nearest-neighbour on propensity score (with replacement)",
            "",
            "  Assumptions",
            "  " + "┄" * 48,
        ]
        for a in MATCHING_ASSUMPTIONS:
            tag = "  testable  " if a.testable else " untestable "
            lines.append(f"  [{tag}]  {a.name}")
        lines.append("")
        return "\n".join(lines)

    def refute(self, data: pd.DataFrame):
        """
        Run refutation checks against this matching estimation.

        Currently runs:

        - **Placebo treatment**: randomly permutes treatment labels and
          re-runs matching. The placebo ATT should be near zero.
        - **Random common cause**: adds a random noise covariate to the
          propensity score model and checks that the ATT is stable.

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``fit()``.
        """
        from ..refutations.matching import (
            MatchingRefutationReport,
            _check_placebo_treatment,
            _check_random_common_cause,
        )
        checks = [
            _check_placebo_treatment(
                data, self._treatment, self._outcome,
                self._adjustment_set, self.effect, self.std_err,
            ),
            _check_random_common_cause(
                data, self._treatment, self._outcome,
                self._adjustment_set, self.effect, self.std_err,
            ),
        ]
        return MatchingRefutationReport(
            checks=checks,
            treatment=self._treatment,
            outcome=self._outcome,
        )

    def __repr__(self) -> str:
        return self.summary()


# ── Estimator ──────────────────────────────────────────────────────────────────

class PropensityScoreMatching:
    """
    Observational estimator using propensity score matching (1-to-1 nearest
    neighbour, with replacement).

    Uses the DAG to identify observed confounders (backdoor criterion), then:

    1. Estimates propensity scores via logistic regression of treatment on
       the adjustment set.
    2. Matches each treated unit to its nearest control by propensity score.
    3. Estimates the ATT as the mean outcome difference across matched pairs.
    4. Computes standard errors via bootstrap over the full procedure.

    Requires **binary treatment** (0/1). Raises ``IdentificationError`` if any
    DAG-declared confounders are absent from the data.

    Example::

        dag = DAG()
        dag.assume("ability").causes("education", "income")
        dag.assume("education").causes("income")

        result = PropensityScoreMatching(
            dag, treatment="education", outcome="income"
        ).fit(df)
        print(result.summary())
    """

    def __init__(self, dag: DAG, treatment: str, outcome: str) -> None:
        self._dag = dag
        self._treatment = treatment
        self._outcome = outcome
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        nodes = self._dag.nodes
        for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
            if var not in nodes:
                raise ValueError(
                    f"{label} '{var}' is not a node in the DAG. "
                    f"Known nodes: {sorted(nodes)}"
                )
        if self._treatment == self._outcome:
            raise ValueError("Treatment and outcome must be different variables.")

    def _identify(self, data_columns: set[str]) -> tuple[set[str], set[str]]:
        dag = self._dag
        T, Y = self._treatment, self._outcome

        treatment_ancestors = dag.ancestors(T)
        outcome_ancestors   = dag.ancestors(Y)
        treatment_descendants = dag.descendants(T)

        confounders = (treatment_ancestors & outcome_ancestors) - treatment_descendants
        observed    = {c for c in confounders if c in data_columns}
        missing     = {c for c in confounders if c not in data_columns}
        return observed, missing

    def fit(self, data: pd.DataFrame) -> MatchingResult:
        """
        Identify confounders, match on propensity scores, and estimate the ATT.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain a binary (0/1) treatment column, an outcome column,
            and any confounders declared in the DAG.

        Raises
        ------
        ``IdentificationError``
            If DAG-declared confounders are absent from the dataframe.
        ``ValueError``
            If treatment or outcome columns are missing, or treatment is not
            binary with both classes present.
        """
        data_columns = set(data.columns)

        for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
            if var not in data_columns:
                raise ValueError(f"{label} column '{var}' not found in dataframe.")

        t_vals = set(data[self._treatment].dropna().unique())
        if not t_vals <= {0, 1, 0.0, 1.0}:
            raise ValueError(
                f"Treatment '{self._treatment}' must be binary (0/1). "
                f"Found values: {sorted(t_vals)}"
            )
        if not ({0, 1} <= {int(v) for v in t_vals}):
            raise ValueError(
                f"Treatment '{self._treatment}' must contain both 0 and 1. "
                f"Found only: {t_vals}"
            )

        adjustment_set, missing = self._identify(data_columns)

        if missing:
            raise IdentificationError(
                f"\nDAG confounders not found in dataframe: {sorted(missing)}\n\n"
                f"Your DAG declares these variables as confounders of "
                f"'{self._treatment}' and '{self._outcome}', but they are "
                f"absent from the dataframe and cannot be controlled for.\n\n"
                f"Consider:\n"
                f"  - Collecting data on {sorted(missing)} and adding it to the dataframe\n"
                f"  - IV estimation if you have a valid instrument for '{self._treatment}'\n"
                f"  - DiD or RD if a natural experiment is available"
            )

        T, Y = self._treatment, self._outcome

        # Unadjusted estimate: naive mean difference, no matching.
        unadjusted = (
            data.loc[data[T] == 1, Y].mean() - data.loc[data[T] == 0, Y].mean()
        )

        # Pre-convert once; reuse numpy arrays throughout.
        T_arr = data[T].values.astype(float)
        Y_arr = data[Y].values.astype(float)

        # Point estimate ATT.
        ps  = _propensity_scores(data, T, adjustment_set)
        att = _att_from_ps(T_arr, Y_arr, ps)

        # Bootstrap SE and CI.
        # Use a minimal DataFrame (treatment + confounders only) to avoid
        # copying unneeded columns on every iteration.
        rng     = np.random.default_rng(_BOOTSTRAP_SEED)
        n       = len(data)
        ps_cols = sorted({T} | adjustment_set)
        ps_data = data[ps_cols]
        boot    = []
        for _ in range(_BOOTSTRAP_N):
            idx = rng.integers(0, n, size=n)
            bd  = ps_data.iloc[idx].reset_index(drop=True)
            try:
                bps = _propensity_scores(bd, T, adjustment_set)
                boot.append(_att_from_ps(T_arr[idx], Y_arr[idx], bps))
            except Exception:
                # Degenerate sample or logit failure — skip this replicate.
                continue

        return MatchingResult(
            att=att,
            unadjusted_effect=float(unadjusted),
            bootstrap_atts=np.array(boot),
            treatment=T,
            outcome=Y,
            adjustment_set=adjustment_set,
            dag=self._dag,
        )
