from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from ..dag import DAG
from ..refutations._check import Assumption

DID_ASSUMPTIONS: list[Assumption] = [
    Assumption(
        "Parallel trends: treated and control groups would have followed "
        "the same trend absent treatment",
        testable=False,
    ),
    Assumption("No anticipation: treatment does not affect outcomes before it begins", testable=False),
    Assumption("Stable group composition: group membership does not change due to treatment", testable=False),
    Assumption("Stable Unit Treatment Value Assumption (SUTVA)", testable=False),
]


class DiDResult:
    """
    The result of a Difference-in-Differences estimation.

    The DiD estimate is the ATT: how much better (or worse) the treated
    group did relative to what would have been expected based on the
    control group's trajectory.
    """

    def __init__(
        self,
        result,
        group: str,
        time: str,
        outcome: str,
        naive_diff: float,
        dag,
    ) -> None:
        self._result = result
        self._group = group
        self._time = time
        self._outcome = outcome
        self._naive_diff = naive_diff
        self._dag = dag

    @property
    def _interaction(self) -> str:
        return f"{self._group}:{self._time}"

    @property
    def effect(self) -> float:
        """DiD point estimate: ATT under the parallel trends assumption."""
        return float(self._result.params[self._interaction])

    @property
    def naive_diff(self) -> float:
        """Naive post-period difference: treated mean minus control mean in the post period."""
        return self._naive_diff

    @property
    def std_err(self) -> float:
        """Standard error of the DiD estimate."""
        return float(self._result.bse[self._interaction])

    @property
    def conf_int(self) -> tuple[float, float]:
        """95% confidence interval for the DiD estimate."""
        ci = self._result.conf_int()
        return (float(ci.loc[self._interaction, 0]), float(ci.loc[self._interaction, 1]))

    @property
    def pvalue(self) -> float:
        """p-value for the DiD estimate (``H0: ATT = 0``)."""
        return float(self._result.pvalues[self._interaction])

    @property
    def statsmodels_result(self):
        """The underlying statsmodels OLS result, for full diagnostics."""
        return self._result

    @property
    def assumptions(self) -> list[Assumption]:
        """Modelling assumptions required for a causal interpretation."""
        return list(DID_ASSUMPTIONS)

    def executive_summary(self) -> str:
        """Narrative explanation of the method, DAG, assumptions, and result."""
        from .._explain import explain_did
        return explain_did(self)

    def summary(self) -> str:
        """Concise tabular summary of the ATT estimate, confidence interval, and assumptions."""
        lo, hi = self.conf_int
        baseline_bias = self._naive_diff - self.effect
        lines = [
            "",
            f"DiD Causal Effect: ({self._group} \u00d7 {self._time}) \u2192 {self._outcome}",
            f"  Estimand: ATT (average treatment effect on the treated)",
            "\u2500" * 54,
            f"  DiD estimate         : {self.effect:>10.4f}",
            f"  Naive post-diff      : {self._naive_diff:>10.4f}  (treated post \u2212 control post)",
            f"  Baseline bias removed: {baseline_bias:>+10.4f}",
            "",
            f"  Std. error           : {self.std_err:>10.4f}",
            f"  95% CI               : [{lo:.4f}, {hi:.4f}]",
            f"  p-value              : {self.pvalue:>10.4f}",
            "",
            "  Assumptions",
            "  " + "\u2504" * 48,
        ]
        for a in DID_ASSUMPTIONS:
            lines.append(f"  {a.fmt_tag()}  {a.name}")
        lines.append("")
        return "\n".join(lines)

    def refute(self, data: pd.DataFrame):
        """
        Run refutation checks against this DiD estimation.

        Currently runs:

        - **Placebo group**: randomly permutes group labels and re-runs DiD.
          The placebo estimate should be near zero if the result is not spurious.
        - **Placebo time**: randomly permutes time labels and re-runs DiD.
          The placebo estimate should be near zero if the effect is genuinely
          concentrated in the post period.
        - **Random common cause**: adds a random noise column as an extra
          control and checks that the estimate does not shift by more than
          one standard error.

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``fit()``.
        """
        from ..refutations.did import (
            DiDRefutationReport,
            _check_placebo_group,
            _check_placebo_time,
            _check_random_common_cause,
        )
        checks = [
            _check_placebo_group(
                data, self._group, self._time, self._outcome,
                self.effect, self.std_err,
            ),
            _check_placebo_time(
                data, self._group, self._time, self._outcome,
                self.effect, self.std_err,
            ),
            _check_random_common_cause(
                data, self._group, self._time, self._outcome,
                self.effect, self.std_err,
            ),
        ]
        return DiDRefutationReport(
            checks=checks,
            group=self._group,
            time=self._time,
            outcome=self._outcome,
        )

    def __repr__(self) -> str:
        return self.summary()


class DiD:
    """
    Difference-in-Differences estimator.

    Estimates the Average Treatment Effect on the Treated (ATT) by comparing
    how outcomes changed over time for the treated group versus the control
    group. The key insight is that any time trend common to both groups
    cancels out, isolating the treatment effect.

    Implemented as OLS with group and time main effects plus their
    interaction::

        outcome ~ group + time + group:time

    The coefficient on ``group:time`` is the DiD estimate.

    The DAG is used to validate that group, time, and outcome are declared
    nodes. It does not apply the backdoor criterion — identification in DiD
    comes from the panel design, not from controlling for observed confounders.

    Requires **binary** group (0 = control, 1 = treated) and time
    (0 = pre-period, 1 = post-period) columns.

    Example::

        dag = DAG()
        dag.assume("group").causes("outcome")
        dag.assume("time").causes("outcome")

        result = DiD(dag, group="group", time="time", outcome="outcome").fit(df)
        print(result.summary())
    """

    def __init__(self, dag: DAG, group: str, time: str, outcome: str) -> None:
        self._dag = dag
        self._group = group
        self._time = time
        self._outcome = outcome
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        nodes = self._dag.nodes
        for label, var in [
            ("Group", self._group),
            ("Time", self._time),
            ("Outcome", self._outcome),
        ]:
            if var not in nodes:
                raise ValueError(
                    f"{label} '{var}' is not a node in the DAG. "
                    f"Known nodes: {sorted(nodes)}"
                )
        if len({self._group, self._time, self._outcome}) < 3:
            raise ValueError("Group, time, and outcome must be different variables.")

    def fit(self, data: pd.DataFrame) -> DiDResult:
        """
        Estimate the ATT via OLS with group and time fixed effects.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain binary (0/1) group and time columns, and a
            numeric outcome column.

        Raises
        ------
        ``ValueError``
            If required columns are missing or group/time are not binary.
        """
        for label, var in [
            ("Group", self._group),
            ("Time", self._time),
            ("Outcome", self._outcome),
        ]:
            if var not in data.columns:
                raise ValueError(f"{label} column '{var}' not found in dataframe.")

        for label, var in [("Group", self._group), ("Time", self._time)]:
            vals = set(data[var].dropna().unique())
            if not vals <= {0, 1, 0.0, 1.0}:
                raise ValueError(
                    f"{label} '{var}' must be binary (0/1). Found: {sorted(vals)}"
                )

        G, T, Y = self._group, self._time, self._outcome

        treated_post = data.loc[(data[G] == 1) & (data[T] == 1), Y].mean()
        control_post = data.loc[(data[G] == 0) & (data[T] == 1), Y].mean()
        naive_diff = float(treated_post - control_post)

        result = smf.ols(f"{Y} ~ {G} * {T}", data=data).fit()

        return DiDResult(result, group=G, time=T, outcome=Y, naive_diff=naive_diff, dag=self._dag)
