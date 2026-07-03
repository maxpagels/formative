from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from .._exceptions import IdentificationError
from ..dag import DAG
from ..refutations._check import Assumption
from ._base import _StatsmodelsResult
from ._cate import (
    CATE_ASSUMPTIONS,
    _CATEResultMixin,
    _fit_cate,
    _validate_modifier_dag,
    _validate_modifier_data,
)

OLS_ASSUMPTIONS: list[Assumption] = [
    Assumption("No unobserved confounders (selection on observables)", testable=False),
    Assumption("Correct functional form for control variables", testable=False),
    Assumption("No reverse causality", testable=False),
    Assumption("No measurement error in key variables", testable=False),
    Assumption("Stable Unit Treatment Value Assumption (SUTVA)", testable=False),
]


class OLSResult(_StatsmodelsResult):
    """
    The result of an OLS causal estimation.

    Holds both the adjusted estimate (controlling for confounders) and the
    unadjusted estimate (``treatment ~ outcome`` only), so you can see the
    effect of controlling for confounders directly.
    """

    _ASSUMPTIONS = OLS_ASSUMPTIONS

    def __init__(
        self,
        adjusted_result,
        unadjusted_result,
        treatment: str,
        outcome: str,
        adjustment_set: set[str],
        dag,
        n: int,
    ) -> None:
        self._result = adjusted_result
        self._unadjusted = unadjusted_result
        self._treatment = treatment
        self._outcome = outcome
        self._adjustment_set = adjustment_set
        self._dag = dag
        self._n = n

    @property
    def unadjusted_effect(self) -> float:
        """Unadjusted point estimate: naive regression without controlling for confounders."""
        return float(self._unadjusted.params[self._treatment])

    @property
    def adjustment_set(self) -> set[str]:
        """Variables controlled for to satisfy the backdoor criterion."""
        return self._adjustment_set

    @property
    def statsmodels_unadjusted_result(self):
        """The underlying unadjusted statsmodels result, for full diagnostics."""
        return self._unadjusted

    def executive_summary(self) -> str:
        """Narrative explanation of the method, DAG, assumptions, and result."""
        from .._explain import explain_ols

        return explain_ols(self)

    def summary(self) -> str:
        """Concise tabular summary of the ATE estimate, confidence interval, and assumptions."""
        lo, hi = self.conf_int
        adj = sorted(self._adjustment_set)
        bias = self.unadjusted_effect - self.effect

        lines = [
            "",
            f"OLS Causal Effect: {self._treatment} → {self._outcome}",
            "─" * 50,
        ]

        if adj:
            lines += [
                f"  Adjusted estimate    : {self.effect:>10.4f}  (controlling for: {', '.join(adj)})",
                f"  Unadjusted estimate  : {self.unadjusted_effect:>10.4f}  (no controls)",
                f"  Confounding bias     : {bias:>+10.4f}",
            ]
        else:
            lines += [
                f"  Estimate             : {self.effect:>10.4f}  (no confounders in DAG)",
            ]

        lines += [
            "",
            f"  Std. error           : {self.std_err:>10.4f}",
            f"  95% CI               : [{lo:.4f}, {hi:.4f}]",
            f"  p-value              : {self.pvalue:>10.4f}",
            f"  N                    : {self._n:>10}",
        ]
        lines += self._extra_summary_lines()
        lines += self._assumptions_lines()
        return "\n".join(lines)

    def refute(self, data: pd.DataFrame):
        """
        Run refutation checks against this OLS estimation.

        Currently runs:

        - **Random common cause**: adds a random noise column as an extra
          control and checks that the estimate does not shift by more than
          one standard error.

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``fit()``.
        """
        from ..refutations.ols import OLSRefutationReport, _check_random_common_cause

        checks = [
            _check_random_common_cause(
                data,
                self._treatment,
                self._outcome,
                self._adjustment_set,
                self.effect,
                self.std_err,
            ),
        ]
        return OLSRefutationReport(
            checks=checks,
            treatment=self._treatment,
            outcome=self._outcome,
        )


class OLSCATEResult(_CATEResultMixin, OLSResult):
    """
    The result of an OLS causal estimation with an effect modifier.

    Everything in ``OLSResult``, plus heterogeneous treatment effects:
    ``effect_by_group`` gives the effect within each modifier level,
    ``homogeneity_pvalue`` tests whether the heterogeneity is real, and
    ``decide_by_group`` produces a cost-benefit decision per group. The
    headline ``effect`` is the sample-share-weighted average of the group
    effects (equivalent to the ATE under the interaction model).
    """

    _ASSUMPTIONS = OLS_ASSUMPTIONS + CATE_ASSUMPTIONS

    def __init__(
        self,
        cate_fit,
        unadjusted_result,
        treatment: str,
        outcome: str,
        modifier: str,
        adjustment_set: set[str],
        dag,
        n: int,
    ) -> None:
        super().__init__(cate_fit.result, unadjusted_result, treatment, outcome, adjustment_set, dag, n)
        self._modifier = modifier
        self._cate = cate_fit

    def refute(self, data: pd.DataFrame):
        """
        Run refutation checks against this heterogeneous-effects estimation.

        Currently runs:

        - **Random common cause**: adds a random noise column as an extra
          control and checks that the weighted average effect does not shift
          by more than one standard error.
        - **Placebo modifier**: randomly permutes the modifier column; the
          heterogeneity should vanish (homogeneity test non-significant).
        - **Random modifier**: interacts treatment with a pure-noise column
          instead; it should show no significant heterogeneity.

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``fit()``.
        """
        from ..refutations.cate import (
            _check_placebo_modifier,
            _check_random_common_cause,
            _check_random_modifier,
        )
        from ..refutations.ols import OLSRefutationReport

        checks = [
            _check_random_common_cause(
                data,
                self._treatment,
                self._outcome,
                self._modifier,
                self._adjustment_set,
                self.effect,
                self.std_err,
            ),
            _check_placebo_modifier(data, self._treatment, self._outcome, self._modifier, self._adjustment_set),
            _check_random_modifier(data, self._treatment, self._outcome, self._modifier, self._adjustment_set),
        ]
        return OLSRefutationReport(
            checks=checks,
            treatment=self._treatment,
            outcome=self._outcome,
        )


class OLSObservational:
    """
    Observational Ordinary Least Squares (OLS) estimator with DAG-based confounder identification.

    Given a DAG encoding your causal assumptions, this estimator:

    1. Identifies which variables must be controlled for (the adjustment set)
       using the backdoor criterion — any variable that is an ancestor of both
       treatment and outcome, but not a descendant of treatment.
    2. Raises an error if unobserved confounders make OLS invalid.
    3. Estimates the causal effect via OLS, controlling for the adjustment set.
    4. Also runs unadjusted OLS so you can see the confounding bias directly.

    Example::

        dag = DAG()
        dag.assume("ability").causes("education", "income")
        dag.assume("education").causes("income")

        # If 'ability' is in df, it is controlled for automatically.
        # If 'ability' is not in df, an IdentificationError is raised.
        result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
        print(result.summary())

    To estimate heterogeneous effects, pass a discrete ``effect_modifier``
    column: the model becomes ``outcome ~ treatment * C(modifier) + controls``
    and ``fit()`` returns an ``OLSCATEResult`` with per-group effects. The
    modifier must be a DAG node that causes the outcome and is **not** a
    descendant of the treatment (conditioning on a mediator manufactures
    artificial heterogeneity — this is validated and raises ``ValueError``).
    """

    def __init__(self, dag: DAG, treatment: str, outcome: str, effect_modifier: str | None = None) -> None:
        self._dag = dag
        self._treatment = treatment
        self._outcome = outcome
        self._modifier = effect_modifier
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        nodes = self._dag.nodes
        for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
            if var not in nodes:
                raise ValueError(f"{label} '{var}' is not a node in the DAG. Known nodes: {sorted(nodes)}")
        if self._treatment == self._outcome:
            raise ValueError("Treatment and outcome must be different variables.")
        if self._modifier is not None:
            _validate_modifier_dag(self._dag, self._treatment, self._outcome, self._modifier)

    def _identify(self, data_columns: set[str]) -> tuple[set[str], set[str]]:
        """
        Apply the backdoor criterion to find the adjustment set.

        A variable Z should be in the adjustment set if it is:
          - An ancestor of both treatment and outcome (a common cause / confounder)
          - Not a descendant of treatment (not a mediator or post-treatment variable)

        Whether a confounder is observed is determined by whether it appears
        in ``data_columns`` — no explicit marking required.

        Returns ``(adjustment_set, dag_confounders_not_in_data)``. The second
        set contains only confounders explicitly modelled in the DAG that are
        absent from the data. There may be additional unobserved confounders
        not represented in the DAG at all.
        """
        dag = self._dag
        T, Y = self._treatment, self._outcome

        treatment_ancestors = dag.ancestors(T)
        outcome_ancestors = dag.ancestors(Y)
        treatment_descendants = dag.descendants(T)

        confounders = (treatment_ancestors & outcome_ancestors) - treatment_descendants

        observed_confounders = {c for c in confounders if c in data_columns}
        dag_confounders_not_in_data = {c for c in confounders if c not in data_columns}

        return observed_confounders, dag_confounders_not_in_data

    def fit(self, data: pd.DataFrame) -> OLSResult:
        """
        Identify the adjustment set, then estimate the causal effect via OLS.

        Runs both an adjusted model (controlling for confounders) and an
        unadjusted model (treatment only) so you can see the confounding bias.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns for treatment, outcome, and any confounders
            that appear in the DAG. Column names must match the node names in
            the DAG. DAG nodes absent from the dataframe are treated as
            unobserved — if any of these are confounders, an ``IdentificationError``
            is raised before estimation.

        Returns
        -------
        OLSResult
            Or an ``OLSCATEResult`` (a subclass adding per-group effects) when
            the estimator was constructed with an ``effect_modifier``.

        Raises
        ------
        ``IdentificationError``
            If confounders declared in the DAG are absent from the dataframe.
            Note: confounders not modelled in the DAG at all cannot be detected.
        ``ValueError``
            If treatment or outcome columns are missing from the dataframe.
        """
        data_columns = set(data.columns)

        for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
            if var not in data_columns:
                raise ValueError(f"{label} column '{var}' not found in dataframe.")

        adjustment_set, dag_confounders_not_in_data = self._identify(data_columns)

        if dag_confounders_not_in_data:
            raise IdentificationError(
                f"\nDAG confounders not found in dataframe: {sorted(dag_confounders_not_in_data)}\n\n"
                f"Your DAG declares these variables as confounders of '{self._treatment}' and\n"
                f"'{self._outcome}', but they are absent from the dataframe and cannot be\n"
                f"controlled for. Note: there may also be confounders not modelled in your\n"
                f"DAG at all — formative cannot detect those.\n\n"
                f"Consider:\n"
                f"  - Collecting data on {sorted(dag_confounders_not_in_data)} and adding it to the dataframe\n"
                f"  - IV estimation if you have a valid instrument for '{self._treatment}'\n"
                f"  - DiD or RD if a natural experiment is available"
            )

        unadjusted_result = smf.ols(f"{self._outcome} ~ {self._treatment}", data=data).fit()

        if self._modifier is not None:
            _validate_modifier_data(data, self._treatment, self._modifier)
            cate_fit = _fit_cate(data, self._treatment, self._outcome, self._modifier, adjustment_set)
            return OLSCATEResult(
                cate_fit,
                unadjusted_result,
                self._treatment,
                self._outcome,
                self._modifier,
                adjustment_set,
                self._dag,
                len(data),
            )

        controls = sorted(adjustment_set)
        rhs = " + ".join([self._treatment] + controls)
        adjusted_result = smf.ols(f"{self._outcome} ~ {rhs}", data=data).fit()

        return OLSResult(
            adjusted_result,
            unadjusted_result,
            self._treatment,
            self._outcome,
            adjustment_set,
            self._dag,
            len(data),
        )
