from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS as _IV2SLS

from ..dag import DAG
from .._exceptions import IdentificationError
from ..refutations._check import Assumption

IV_ASSUMPTIONS: list[Assumption] = [
    Assumption("Relevance: the instrument strongly affects treatment", testable=True),
    Assumption("Exclusion restriction: instrument only affects outcome through treatment", testable=False),
    Assumption("Independence: instrument is uncorrelated with unobserved confounders", testable=False),
    Assumption("Monotonicity: instrument affects treatment in same direction for everyone", testable=False),
]


class IVResult:
    """
    The result of an IV (2SLS) causal estimation.

    Holds both the instrumented (2SLS) estimate and the unadjusted OLS
    estimate (``outcome ~ treatment`` only), so you can see the confounding
    bias that IV corrects.
    """

    def __init__(
        self,
        result,
        unadjusted_result,
        treatment: str,
        outcome: str,
        instrument: str,
        adjustment_set: set[str],
    ) -> None:
        self._result = result
        self._unadjusted = unadjusted_result
        self._treatment = treatment
        self._outcome = outcome
        self._instrument = instrument
        self._adjustment_set = adjustment_set

    @property
    def effect(self) -> float:
        """2SLS point estimate of the causal effect of treatment on outcome."""
        return float(self._result.params[self._treatment])

    @property
    def unadjusted_effect(self) -> float:
        """Unadjusted OLS estimate: naive regression without instrument or controls."""
        return float(self._unadjusted.params[self._treatment])

    @property
    def std_err(self) -> float:
        """Standard error of the 2SLS treatment effect estimate."""
        return float(self._result.bse[self._treatment])

    @property
    def conf_int(self) -> tuple[float, float]:
        """95% confidence interval for the 2SLS treatment effect."""
        ci = self._result.conf_int()
        return (float(ci.loc[self._treatment, 0]), float(ci.loc[self._treatment, 1]))

    @property
    def pvalue(self) -> float:
        """p-value for the 2SLS treatment effect (``H0: effect = 0``)."""
        return float(self._result.pvalues[self._treatment])

    @property
    def adjustment_set(self) -> set[str]:
        """Observed confounders included as controls in both OLS stages."""
        return self._adjustment_set

    @property
    def statsmodels_result(self):
        """The underlying statsmodels IV2SLS result, for full diagnostics."""
        return self._result

    @property
    def statsmodels_unadjusted_result(self):
        """The underlying unadjusted OLS result, for full diagnostics."""
        return self._unadjusted

    @property
    def assumptions(self) -> list[Assumption]:
        """Modelling assumptions required for a causal interpretation."""
        return list(IV_ASSUMPTIONS)

    def refute(self, data: pd.DataFrame):
        """
        Run refutation checks against this IV estimation.

        Re-uses the original data to run statistical tests that probe the
        assumptions underlying the IV estimate. Returns an
        ``IVRefutationReport`` with one ``RefutationCheck`` per test.

        Currently runs:

        - **First-stage F-statistic**: tests instrument relevance.
          ``F < 10`` indicates a weak instrument (Stock & Yogo, 2005).

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``fit()``.
        """
        from ..refutations.iv import (
            IVRefutationReport,
            _check_first_stage_f,
            _check_random_common_cause,
        )

        controls = sorted(self._adjustment_set)
        checks = [
            _check_first_stage_f(data, self._treatment, self._instrument, controls),
            _check_random_common_cause(
                data, self._treatment, self._outcome, self._instrument,
                self._adjustment_set, self.effect, self.std_err,
            ),
        ]
        return IVRefutationReport(
            checks=checks,
            treatment=self._treatment,
            outcome=self._outcome,
            instrument=self._instrument,
        )

    def summary(self) -> str:
        lo, hi = self.conf_int
        adj = sorted(self._adjustment_set)
        bias = self.unadjusted_effect - self.effect
        controls_note = f"  (controlling for: {', '.join(adj)})" if adj else ""
        lines = [
            "",
            f"IV (2SLS) Causal Effect: {self._treatment} → {self._outcome}",
            f"  Instrument: {self._instrument}",
            "─" * 50,
            f"  IV estimate          : {self.effect:>10.4f}{controls_note}",
            f"  Unadjusted estimate  : {self.unadjusted_effect:>10.4f}  (no controls)",
            f"  Confounding bias     : {bias:>+10.4f}",
            "",
            f"  Std. error           : {self.std_err:>10.4f}",
            f"  95% CI               : [{lo:.4f}, {hi:.4f}]",
            f"  p-value              : {self.pvalue:>10.4f}",
            "",
            "  Assumptions",
            "  " + "┄" * 48,
        ]
        for a in IV_ASSUMPTIONS:
            tag = "  testable  " if a.testable else " untestable "
            lines.append(f"  [{tag}]  {a.name}")
        lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class IV2SLS:
    """
    Instrumental Variables estimator using Two-Stage Least Squares (2SLS).

    Uses the DAG to validate the instrument structurally and identify
    observed confounders to include as controls in both stages:

    1. **Relevance** (structural): the instrument must have a directed path
       to the treatment in the DAG.
    2. **Exclusion restriction** (structural): no directed path from the
       instrument to the outcome that bypasses the treatment.
    3. **Observed confounders** (backdoor criterion): any variable that is
       a common cause of treatment and outcome and is present in the data
       is included as a control. Unobserved confounders are handled by the
       instrument and do **not** raise an ``IdentificationError`` — this is
       the primary use case for IV estimation.

    Example::

        dag = DAG()
        dag.assume("proximity").causes("education")
        dag.assume("ability").causes("education", "income")
        dag.assume("education").causes("income")

        # 'ability' is absent from df (unobserved) — the instrument "controls" for it.
        result = IV2SLS(
            dag, treatment="education", outcome="income", instrument="proximity"
        ).fit(df)
        print(result.summary())
    """

    def __init__(
        self, dag: DAG, treatment: str, outcome: str, instrument: str
    ) -> None:
        self._dag = dag
        self._treatment = treatment
        self._outcome = outcome
        self._instrument = instrument
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        dag = self._dag
        T, Y, Z = self._treatment, self._outcome, self._instrument
        nodes = dag.nodes

        for label, var in [("Treatment", T), ("Outcome", Y), ("Instrument", Z)]:
            if var not in nodes:
                raise ValueError(
                    f"{label} '{var}' is not a node in the DAG. "
                    f"Known nodes: {sorted(nodes)}"
                )

        if T == Y:
            raise ValueError("Treatment and outcome must be different variables.")
        if Z == T:
            raise ValueError("Instrument and treatment must be different variables.")
        if Z == Y:
            raise ValueError("Instrument and outcome must be different variables.")

        # Relevance: instrument must have a directed path to treatment.
        if T not in dag.descendants(Z):
            raise ValueError(
                f"Instrument '{Z}' does not cause treatment '{T}' in the DAG "
                f"(no directed path from '{Z}' to '{T}'). "
                f"Add a causal path to assert relevance."
            )

        # Exclusion restriction: removing treatment, instrument must not reach outcome.
        descendants_without_T = self._descendants_excluding_node(Z, T)
        if Y in descendants_without_T:
            raise ValueError(
                f"Exclusion restriction violated: '{Z}' can reach '{Y}' in the DAG "
                f"without going through '{T}'. The instrument must affect the "
                f"outcome only through the treatment."
            )

    def _descendants_excluding_node(self, start: str, excluded: str) -> set[str]:
        """All directed descendants of start, not passing through excluded."""
        dag = self._dag
        result: set[str] = set()
        queue = [c for c in dag.children(start) if c != excluded]
        while queue:
            node = queue.pop()
            if node not in result:
                result.add(node)
                queue.extend(c for c in dag.children(node) if c != excluded)
        return result

    def _identify(self, data_columns: set[str]) -> set[str]:
        """
        Identify observed confounders to include as controls (backdoor criterion).

        Unobserved confounders are handled by the instrument and are not flagged
        as errors — this is the primary use case for IV estimation.
        """
        dag = self._dag
        T, Y = self._treatment, self._outcome

        treatment_ancestors = dag.ancestors(T)
        outcome_ancestors = dag.ancestors(Y)
        treatment_descendants = dag.descendants(T)

        confounders = (treatment_ancestors & outcome_ancestors) - treatment_descendants
        confounders.discard(self._instrument)

        return {c for c in confounders if c in data_columns}

    def fit(self, data: pd.DataFrame) -> IVResult:
        """
        Validate data and estimate the causal effect via 2SLS.

        Observed confounders from the DAG are included as controls in both
        stages. Unobserved confounders are handled by the instrument.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns for treatment, outcome, and instrument.
            Observed confounders present in the DAG are added as controls
            automatically if they appear in the dataframe.

        Raises
        ------
        ``ValueError``
            If treatment, outcome, or instrument columns are missing from
            the dataframe.
        """
        data_columns = set(data.columns)

        for label, var in [
            ("Treatment", self._treatment),
            ("Outcome", self._outcome),
            ("Instrument", self._instrument),
        ]:
            if var not in data_columns:
                raise ValueError(f"{label} column '{var}' not found in dataframe.")

        adjustment_set = self._identify(data_columns)
        controls = sorted(adjustment_set)

        T, Y, Z = self._treatment, self._outcome, self._instrument

        # Build design matrices for 2SLS:
        #   exog:       [const, T, controls]  — includes endogenous treatment
        #   instrument: [const, Z, controls]  — instrument replaces T in position 1
        # Column names are aligned so statsmodels indexes result params by T's name.
        X = sm.add_constant(data[[T] + controls], prepend=True)
        Z_mat = sm.add_constant(data[[Z] + controls], prepend=True)
        Z_mat.columns = X.columns  # rename Z column to T so params are indexed by T

        model = _IV2SLS(endog=data[Y], exog=X, instrument=Z_mat)
        result = model.fit()
        unadjusted_result = smf.ols(f"{Y} ~ {T}", data=data).fit()

        return IVResult(result, unadjusted_result, T, Y, Z, adjustment_set)
