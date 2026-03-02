from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from ..dag import DAG
from ..refutations._check import Assumption

RCT_ASSUMPTIONS: list[Assumption] = [
    Assumption("Random assignment of treatment", testable=False),
    Assumption("Excludability: assignment affects outcome only through treatment received", testable=False),
    Assumption("Stable Unit Treatment Value Assumption (SUTVA)", testable=False),
]


class RCTResult:
    """
    The result of an RCT causal estimation.

    Estimates the Average Treatment Effect (ATE) via OLS. Because treatment
    is randomly assigned, no confounder adjustment is needed and the ATE
    equals the difference in mean outcomes between treatment and control.
    """

    def __init__(
        self,
        result,
        treatment: str,
        outcome: str,
        dag,
    ) -> None:
        self._result = result
        self._treatment = treatment
        self._outcome = outcome
        self._dag = dag

    @property
    def effect(self) -> float:
        """ATE: average treatment effect (difference in means)."""
        return float(self._result.params[self._treatment])

    @property
    def std_err(self) -> float:
        """Standard error of the ATE estimate."""
        return float(self._result.bse[self._treatment])

    @property
    def conf_int(self) -> tuple[float, float]:
        """95% confidence interval for the ATE."""
        ci = self._result.conf_int()
        return (float(ci.loc[self._treatment, 0]), float(ci.loc[self._treatment, 1]))

    @property
    def pvalue(self) -> float:
        """p-value for the ATE (``H0: ATE = 0``)."""
        return float(self._result.pvalues[self._treatment])

    @property
    def statsmodels_result(self):
        """The underlying statsmodels OLS result, for full diagnostics."""
        return self._result

    @property
    def assumptions(self) -> list[Assumption]:
        """Modelling assumptions required for a causal interpretation."""
        return list(RCT_ASSUMPTIONS)

    def executive_summary(self) -> str:
        """Narrative explanation of the method, DAG, assumptions, and result."""
        from .._explain import explain_rct
        return explain_rct(self)

    def summary(self) -> str:
        """Concise tabular summary of the ATE estimate, confidence interval, and assumptions."""
        lo, hi = self.conf_int
        lines = [
            "",
            f"RCT Causal Effect: {self._treatment} → {self._outcome}",
            f"  Estimand: ATE (average treatment effect)",
            "─" * 50,
            f"  ATE estimate         : {self.effect:>10.4f}",
            "",
            f"  Std. error           : {self.std_err:>10.4f}",
            f"  95% CI               : [{lo:.4f}, {hi:.4f}]",
            f"  p-value              : {self.pvalue:>10.4f}",
            "",
            "  Assumptions",
            "  " + "┄" * 48,
        ]
        for a in RCT_ASSUMPTIONS:
            tag = "  testable  " if a.testable else " untestable "
            lines.append(f"  [{tag}]  {a.name}")
        lines.append("")
        return "\n".join(lines)

    def refute(self, data: pd.DataFrame):
        """
        Run refutation checks against this RCT estimation.

        Currently runs:

        - **Random common cause**: adds a random noise column as an extra
          control and checks that the ATE does not shift by more than one
          standard error. Under randomisation the ATE should be robust to
          any additional covariate.

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``fit()``.
        """
        from ..refutations.rct import RCTRefutationReport, _check_random_common_cause

        checks = [
            _check_random_common_cause(
                data, self._treatment, self._outcome,
                self.effect, self.std_err,
            ),
        ]
        return RCTRefutationReport(
            checks=checks,
            treatment=self._treatment,
            outcome=self._outcome,
        )

    def __repr__(self) -> str:
        return self.summary()


class RCT:
    """
    Randomized Controlled Trial estimator.

    Estimates the Average Treatment Effect (ATE) via OLS regression of
    the outcome on the treatment indicator. Because treatment is randomly
    assigned, no confounder adjustment is needed.

    DAG validation enforces the RCT assumption: treatment must have no
    declared causes (parents) in the DAG. Declaring a cause of treatment
    would contradict random assignment and raises a ``ValueError``.

    Example::

        dag = DAG()
        dag.assume("treatment").causes("outcome")

        result = RCT(dag, treatment="treatment", outcome="outcome").fit(df)
        print(result.summary())
    """

    def __init__(self, dag: DAG, treatment: str, outcome: str) -> None:
        self._dag = dag
        self._treatment = treatment
        self._outcome = outcome
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        dag = self._dag
        nodes = dag.nodes
        T, Y = self._treatment, self._outcome

        for label, var in [("Treatment", T), ("Outcome", Y)]:
            if var not in nodes:
                raise ValueError(
                    f"{label} '{var}' is not a node in the DAG. "
                    f"Known nodes: {sorted(nodes)}"
                )
        if T == Y:
            raise ValueError("Treatment and outcome must be different variables.")

        parents_of_treatment = dag.parents(T)
        if parents_of_treatment:
            raise ValueError(
                f"In an RCT, treatment is randomly assigned and must have no causes "
                f"in the DAG. '{T}' has declared causes: {sorted(parents_of_treatment)}. "
                f"Remove these edges, or use OLSObservational / PropensityScoreMatching "
                f"if treatment is not randomised."
            )

    def fit(self, data: pd.DataFrame) -> RCTResult:
        """
        Estimate the ATE via OLS regression of outcome on treatment.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns for treatment and outcome. Treatment may
            be binary (0/1) or continuous.

        Raises
        ------
        ``ValueError``
            If treatment or outcome columns are missing from the dataframe.
        """
        for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
            if var not in data.columns:
                raise ValueError(f"{label} column '{var}' not found in dataframe.")

        result = smf.ols(f"{self._outcome} ~ {self._treatment}", data=data).fit()
        return RCTResult(result, self._treatment, self._outcome, self._dag)
