from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from ..dag import DAG
from .._exceptions import IdentificationError


class OLSResult:
    """
    The result of an OLS causal estimation.

    Holds both the adjusted estimate (controlling for confounders) and the
    unadjusted estimate (treatment ~ outcome only), so you can see the
    effect of controlling for confounders directly.
    """

    def __init__(
        self,
        adjusted_result,
        unadjusted_result,
        treatment: str,
        outcome: str,
        adjustment_set: set[str],
    ) -> None:
        self._adjusted = adjusted_result
        self._unadjusted = unadjusted_result
        self._treatment = treatment
        self._outcome = outcome
        self._adjustment_set = adjustment_set

    @property
    def effect(self) -> float:
        """Adjusted point estimate: causal effect of treatment on outcome."""
        return float(self._adjusted.params[self._treatment])

    @property
    def unadjusted_effect(self) -> float:
        """Unadjusted point estimate: naive regression without controlling for confounders."""
        return float(self._unadjusted.params[self._treatment])

    @property
    def std_err(self) -> float:
        """Standard error of the adjusted treatment effect estimate."""
        return float(self._adjusted.bse[self._treatment])

    @property
    def conf_int(self) -> tuple[float, float]:
        """95% confidence interval for the adjusted treatment effect."""
        ci = self._adjusted.conf_int()
        return (float(ci.loc[self._treatment, 0]), float(ci.loc[self._treatment, 1]))

    @property
    def pvalue(self) -> float:
        """p-value for the adjusted treatment effect (H0: effect = 0)."""
        return float(self._adjusted.pvalues[self._treatment])

    @property
    def adjustment_set(self) -> set[str]:
        """Variables controlled for to satisfy the backdoor criterion."""
        return self._adjustment_set

    @property
    def statsmodels_result(self):
        """The underlying adjusted statsmodels result, for full diagnostics."""
        return self._adjusted

    @property
    def statsmodels_unadjusted_result(self):
        """The underlying unadjusted statsmodels result, for full diagnostics."""
        return self._unadjusted

    def summary(self) -> str:
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
            "",
            "  Interpretation assumes the DAG correctly captures all",
            "  confounding. Unmodelled confounders will bias this estimate.",
            "",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class OLSObservational:
    """
    Observational OLS estimator with DAG-based confounder identification.

    Given a DAG encoding your causal assumptions, this estimator:
      1. Identifies which variables must be controlled for (the adjustment set)
         using the backdoor criterion.
      2. Raises an error if unobserved confounders make OLS invalid.
      3. Estimates the causal effect via OLS, controlling for the adjustment set.
      4. Also runs unadjusted OLS so you can see the confounding bias directly.

    Usage
    -----
        dag = DAG()
        dag.add_edge("ability", "education")
        dag.add_edge("ability", "income")
        dag.add_edge("education", "income")

        # If 'ability' is in df, it is controlled for automatically.
        # If 'ability' is not in df, an IdentificationError is raised.
        result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
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
        """
        Apply the backdoor criterion to find the adjustment set.

        A variable Z should be in the adjustment set if it is:
          - An ancestor of both treatment and outcome (a common cause / confounder)
          - Not a descendant of treatment (not a mediator or post-treatment variable)

        Whether a confounder is observed is determined by whether it appears
        in data_columns — no explicit marking required.

        Returns (observed_adjustment_set, unobserved_confounders).
        """
        dag = self._dag
        T, Y = self._treatment, self._outcome

        treatment_ancestors = dag.ancestors(T)
        outcome_ancestors = dag.ancestors(Y)
        treatment_descendants = dag.descendants(T)

        confounders = (treatment_ancestors & outcome_ancestors) - treatment_descendants

        observed_confounders = {c for c in confounders if c in data_columns}
        unobserved_confounders = {c for c in confounders if c not in data_columns}

        return observed_confounders, unobserved_confounders

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
            unobserved — if any of these are confounders, an IdentificationError
            is raised before estimation.

        Raises
        ------
        IdentificationError
            If confounders in the DAG are absent from the dataframe and cannot
            be controlled for, making OLS biased.
        ValueError
            If treatment or outcome columns are missing from the dataframe.
        """
        data_columns = set(data.columns)

        for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
            if var not in data_columns:
                raise ValueError(f"{label} column '{var}' not found in dataframe.")

        adjustment_set, unobserved_confounders = self._identify(data_columns)

        if unobserved_confounders:
            raise IdentificationError(
                f"\nUnobserved confounders detected: {sorted(unobserved_confounders)}\n\n"
                f"These variables influence both '{self._treatment}' and '{self._outcome}'\n"
                f"but are not in the dataframe and cannot be controlled for.\n\n"
                f"Consider:\n"
                f"  - IV estimation if you have a valid instrument for '{self._treatment}'\n"
                f"  - DiD or RD if a natural experiment is available\n"
                f"  - Collecting data on {sorted(unobserved_confounders)} and adding it to the dataframe"
            )

        controls = sorted(adjustment_set)
        rhs = " + ".join([self._treatment] + controls)
        adjusted_result = smf.ols(f"{self._outcome} ~ {rhs}", data=data).fit()
        unadjusted_result = smf.ols(f"{self._outcome} ~ {self._treatment}", data=data).fit()

        return OLSResult(
            adjusted_result,
            unadjusted_result,
            self._treatment,
            self._outcome,
            adjustment_set,
        )
