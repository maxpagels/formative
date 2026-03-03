from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from ..dag import DAG
from ..refutations._check import Assumption

RDD_ASSUMPTIONS: list[Assumption] = [
    Assumption(
        "Continuity: potential outcomes are continuous in the running variable at the cutoff",
        testable=False,
    ),
    Assumption(
        "No manipulation: units cannot precisely control which side of the cutoff they are on",
        testable=False,
    ),
    Assumption(
        "Local effect: the estimated LATE at the cutoff applies only to units near the threshold",
        testable=False,
    ),
    Assumption("Stable Unit Treatment Value Assumption (SUTVA)", testable=False),
]


class RDDResult:
    """
    The result of a Regression Discontinuity Design estimation.

    The RDD estimate is the LATE at the cutoff: the jump in the outcome at
    the cutoff, estimated via local linear regression on both sides of the threshold.
    """

    def __init__(
        self,
        result,
        treatment: str,
        running_var: str,
        cutoff: float,
        outcome: str,
        bandwidth: float | None,
        unadjusted_effect: float,
        dag,
    ) -> None:
        self._result = result
        self._treatment = treatment
        self._running_var = running_var
        self._cutoff = cutoff
        self._outcome = outcome
        self._bandwidth = bandwidth
        self._unadjusted_effect = unadjusted_effect
        self._dag = dag

    @property
    def effect(self) -> float:
        """LATE at the cutoff: the jump in outcome at the threshold (coefficient on the treatment indicator)."""
        return float(self._result.params[self._treatment])

    @property
    def unadjusted_effect(self) -> float:
        """Naive mean difference: above-cutoff mean minus below-cutoff mean."""
        return self._unadjusted_effect

    @property
    def std_err(self) -> float:
        """Standard error of the LATE at the cutoff estimate."""
        return float(self._result.bse[self._treatment])

    @property
    def conf_int(self) -> tuple[float, float]:
        """95% confidence interval for the LATE at the cutoff estimate."""
        ci = self._result.conf_int()
        return (float(ci.loc[self._treatment, 0]), float(ci.loc[self._treatment, 1]))

    @property
    def pvalue(self) -> float:
        """p-value for the LATE at the cutoff (``H0: LATE at cutoff = 0``)."""
        return float(self._result.pvalues[self._treatment])

    @property
    def statsmodels_result(self):
        """The underlying statsmodels OLS result, for full diagnostics."""
        return self._result

    @property
    def assumptions(self) -> list[Assumption]:
        """Modelling assumptions required for a causal interpretation."""
        return list(RDD_ASSUMPTIONS)

    @property
    def cutoff(self) -> float:
        """The threshold value that determines treatment assignment."""
        return self._cutoff

    @property
    def running_var(self) -> str:
        """Name of the running variable column."""
        return self._running_var

    @property
    def bandwidth(self) -> float | None:
        """Bandwidth used to restrict observations around the cutoff, or None."""
        return self._bandwidth

    @property
    def n_obs(self) -> int:
        """Number of observations used in the estimation (after bandwidth filtering)."""
        return int(self._result.nobs)

    def executive_summary(self) -> str:
        """Narrative explanation of the method, DAG, assumptions, and result."""
        from .._explain import explain_rdd
        return explain_rdd(self)

    def summary(self) -> str:
        """Concise tabular summary of the LATE at the cutoff, confidence interval, and assumptions."""
        lo, hi = self.conf_int
        bw_label = f"{self._bandwidth:.4f}" if self._bandwidth is not None else "all data"
        lines = [
            "",
            f"RDD Causal Effect: {self._running_var} \u2192 {self._treatment} \u2192 {self._outcome}  |  cutoff: {self._cutoff:.4f}",
            f"  Estimand: LATE at the cutoff",
            f"  Bandwidth: {bw_label}",
            "\u2500" * 54,
            f"  LATE at cutoff        : {self.effect:>10.4f}",
            f"  Unadjusted mean diff  : {self._unadjusted_effect:>10.4f}  (above-cutoff \u2212 below-cutoff mean)",
            "",
            f"  Std. error            : {self.std_err:>10.4f}",
            f"  95% CI                : [{lo:.4f}, {hi:.4f}]",
            f"  p-value               : {self.pvalue:>10.4f}",
            f"  N (in bandwidth)      : {self.n_obs:>9}",
            "",
            "  Assumptions",
            "  " + "\u2504" * 30,
        ]
        for a in RDD_ASSUMPTIONS:
            lines.append(f"  {a.fmt_tag()}  {a.name}")
        lines.append("")
        return "\n".join(lines)

    def refute(self, data: pd.DataFrame):
        """
        Run refutation checks against this RDD estimation.

        Currently runs:

        - **Placebo cutoff**: fits an RDD at a false cutoff in the all-control
          region. The placebo estimate should be near zero if the original
          result is not spurious.
        - **Random common cause**: adds a random noise column as an extra
          covariate and checks that the estimate does not shift by more than
          one standard error.

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``fit()``.
        """
        from ..refutations.rdd import (
            RDDRefutationReport,
            _check_placebo_cutoff,
            _check_random_common_cause,
        )
        checks = [
            _check_placebo_cutoff(
                data, self._running_var, self._treatment, self._outcome,
                self._cutoff,
            ),
            _check_random_common_cause(
                data, self._running_var, self._treatment, self._outcome,
                self._cutoff, self.effect, self.std_err,
            ),
        ]
        return RDDRefutationReport(
            checks=checks,
            treatment=self._treatment,
            running_var=self._running_var,
            cutoff=self._cutoff,
            outcome=self._outcome,
        )

    def __repr__(self) -> str:
        return self.summary()


class RDD:
    """
    Regression Discontinuity Design estimator.

    Estimates the Local Average Treatment Effect at the cutoff (LATE at the cutoff)
    by fitting a local linear regression on both sides of the cutoff.
    Treatment is always derived from whether the running variable is at
    or above the cutoff — any existing treatment column in the data is
    overwritten by this rule.

    The model fitted is::

        outcome ~ treatment + (running_var - cutoff) + treatment:(running_var - cutoff)

    The coefficient on ``treatment`` is the LATE at the cutoff: the jump in
    outcome at the threshold, after allowing slopes to differ on each side.

    The DAG is used to validate that the running variable is an ancestor
    of treatment (i.e. the threshold rule is part of the assumed causal
    structure). Identification does not rely on the backdoor criterion —
    it comes from the sharp discontinuity in treatment assignment at the
    cutoff.

    Example::

        dag = DAG()
        dag.assume("score").causes("treatment", "outcome")
        dag.assume("treatment").causes("outcome")

        result = RDD(dag, treatment="treatment", running_var="score",
                     cutoff=0.0, outcome="outcome").fit(df)
        print(result.summary())
    """

    def __init__(
        self,
        dag: DAG,
        treatment: str,
        running_var: str,
        cutoff: float,
        outcome: str,
        bandwidth: float | None = None,
    ) -> None:
        self._dag = dag
        self._treatment = treatment
        self._running_var = running_var
        self._cutoff = cutoff
        self._outcome = outcome
        self._bandwidth = bandwidth
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        nodes = self._dag.nodes
        for label, var in [
            ("Running variable", self._running_var),
            ("Treatment", self._treatment),
            ("Outcome", self._outcome),
        ]:
            if var not in nodes:
                raise ValueError(
                    f"{label} '{var}' is not a node in the DAG. "
                    f"Known nodes: {sorted(nodes)}"
                )
        if len({self._running_var, self._treatment, self._outcome}) < 3:
            raise ValueError(
                "Running variable, treatment, and outcome must be different variables."
            )
        if self._running_var not in self._dag.ancestors(self._treatment):
            raise ValueError(
                f"Running variable '{self._running_var}' must be an ancestor of "
                f"treatment '{self._treatment}' in the DAG."
            )

    def fit(self, data: pd.DataFrame) -> RDDResult:
        """
        Estimate the LATE at the cutoff via local linear regression.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain the running variable and outcome columns.
            The treatment column is derived from the running variable
            and the cutoff — any existing column with the treatment name
            is overwritten.

        Raises
        ------
        ``ValueError``
            If required columns are missing from the dataframe.
        """
        for label, var in [
            ("Running variable", self._running_var),
            ("Outcome", self._outcome),
        ]:
            if var not in data.columns:
                raise ValueError(f"{label} column '{var}' not found in dataframe.")

        # Apply bandwidth filter
        if self._bandwidth is not None:
            mask = (data[self._running_var] - self._cutoff).abs() <= self._bandwidth
            filtered = data.loc[mask].copy()
        else:
            filtered = data.copy()

        # Unadjusted effect: naive mean difference above vs below cutoff
        above = filtered.loc[filtered[self._running_var] >= self._cutoff, self._outcome].mean()
        below = filtered.loc[filtered[self._running_var] < self._cutoff, self._outcome].mean()
        unadjusted_effect = float(above - below)

        # Build working copy with internal regression columns
        work_data = filtered.copy()
        work_data["_rdd_r"] = work_data[self._running_var] - self._cutoff
        work_data[self._treatment] = (work_data[self._running_var] >= self._cutoff).astype(float)

        T, Y = self._treatment, self._outcome
        result = smf.ols(f"{Y} ~ {T} + _rdd_r + {T}:_rdd_r", data=work_data).fit()

        return RDDResult(
            result=result,
            treatment=T,
            running_var=self._running_var,
            cutoff=self._cutoff,
            outcome=Y,
            bandwidth=self._bandwidth,
            unadjusted_effect=unadjusted_effect,
            dag=self._dag,
        )
