from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from .._assumptions import Assumption
from ..dag import DAG
from ._base import _StatsmodelsResult
from ._cate import (
    CATE_ASSUMPTIONS,
    _CATEResultMixin,
    _fit_cate,
    _validate_modifier_dag,
    _validate_modifier_data,
)

RCT_ASSUMPTIONS: list[Assumption] = [
    Assumption("Random assignment of treatment", testable=False),
    Assumption("Excludability: assignment affects outcome only through treatment received", testable=False),
    Assumption("Stable Unit Treatment Value Assumption (SUTVA)", testable=False),
]


class RCTResult(_StatsmodelsResult):
    """
    The result of an RCT causal estimation.

    Estimates the Average Treatment Effect (ATE) via OLS. Because treatment
    is randomly assigned, no confounder adjustment is needed and the ATE
    equals the difference in mean outcomes between treatment and control.
    """

    _ASSUMPTIONS = RCT_ASSUMPTIONS

    def __init__(
        self,
        result,
        treatment: str,
        outcome: str,
        dag,
        n: int,
    ) -> None:
        self._result = result
        self._treatment = treatment
        self._outcome = outcome
        self._dag = dag
        self._n = n

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
            "  Estimand: ATE (average treatment effect)",
            "─" * 50,
            f"  ATE estimate         : {self.effect:>10.4f}",
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
                data,
                self._treatment,
                self._outcome,
                self.effect,
                self.std_err,
            ),
        ]
        return RCTRefutationReport(
            checks=checks,
            treatment=self._treatment,
            outcome=self._outcome,
        )

    def learn_policy(
        self,
        data: pd.DataFrame,
        modifiers: list[str],
        cost: float,
        benefit: float,
        max_depth: int = 2,
    ):
        """
        Learn a treatment assignment policy from this experiment.

        Where ``decide()`` asks *should we treat everyone?*, this asks *whom
        should we treat?* — it searches for the shallow decision tree over the
        candidate features that maximises total net benefit
        (``benefit × effect − cost`` per treated unit), using cross-fitted
        doubly robust scores (Athey & Wager style policy learning).

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``fit()``. Treatment must be
            binary 0/1.
        modifiers : list[str]
            Candidate discrete, pre-treatment feature columns the tree may
            split on. Each must be a DAG node that causes the outcome and is
            not a descendant of the treatment. Bin continuous features first.
        cost : float
            Cost per unit of treatment applied.
        benefit : float
            Benefit (revenue, utility, etc.) per unit increase in the outcome.
        max_depth : int
            Tree depth, 1 or 2. Depth is the regularizer: compare the
            ``value`` of a depth-1 and depth-2 policy to see whether the
            extra complexity earns its keep.

        Returns
        -------
        PolicyResult
            The learned rule, an honest estimate of its per-unit value over
            the best constant policy, and an ``assign()`` method for new data.
        """
        from .policy import _learn_policy

        return _learn_policy(data, self._treatment, self._outcome, modifiers, cost, benefit, max_depth, self._dag)


class RCTCATEResult(_CATEResultMixin, RCTResult):
    """
    The result of an RCT causal estimation with an effect modifier.

    Everything in ``RCTResult``, plus heterogeneous treatment effects:
    ``effect_by_group`` gives the effect within each modifier level,
    ``homogeneity_pvalue`` tests whether the heterogeneity is real, and
    ``decide_by_group`` produces a cost-benefit decision per group. The
    headline ``effect`` is the sample-share-weighted average of the group
    effects (equivalent to the ATE under the interaction model).
    """

    _ASSUMPTIONS = RCT_ASSUMPTIONS + CATE_ASSUMPTIONS

    def __init__(
        self,
        cate_fit,
        treatment: str,
        outcome: str,
        modifier: str,
        dag,
        n: int,
    ) -> None:
        super().__init__(cate_fit.result, treatment, outcome, dag, n)
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
        from ..refutations.rct import RCTRefutationReport

        checks = [
            _check_random_common_cause(
                data,
                self._treatment,
                self._outcome,
                self._modifier,
                set(),
                self.effect,
                self.std_err,
            ),
            _check_placebo_modifier(data, self._treatment, self._outcome, self._modifier, set()),
            _check_random_modifier(data, self._treatment, self._outcome, self._modifier, set()),
        ]
        return RCTRefutationReport(
            checks=checks,
            treatment=self._treatment,
            outcome=self._outcome,
        )


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

    To estimate heterogeneous effects, pass a discrete ``effect_modifier``
    column: the model becomes ``outcome ~ treatment * C(modifier)`` and
    ``fit()`` returns an ``RCTCATEResult`` with per-group effects. The
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
        dag = self._dag
        nodes = dag.nodes
        T, Y = self._treatment, self._outcome

        for label, var in [("Treatment", T), ("Outcome", Y)]:
            if var not in nodes:
                raise ValueError(f"{label} '{var}' is not a node in the DAG. Known nodes: {sorted(nodes)}")
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

        if self._modifier is not None:
            _validate_modifier_dag(dag, T, Y, self._modifier)

    def fit(self, data: pd.DataFrame) -> RCTResult:
        """
        Estimate the ATE via OLS regression of outcome on treatment.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns for treatment and outcome. Treatment may
            be binary (0/1) or continuous.

        Returns
        -------
        RCTResult
            Or an ``RCTCATEResult`` (a subclass adding per-group effects) when
            the estimator was constructed with an ``effect_modifier``.

        Raises
        ------
        ``ValueError``
            If treatment or outcome columns are missing from the dataframe.
        """
        for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
            if var not in data.columns:
                raise ValueError(f"{label} column '{var}' not found in dataframe.")

        if self._modifier is not None:
            _validate_modifier_data(data, self._treatment, self._modifier)
            cate_fit = _fit_cate(data, self._treatment, self._outcome, self._modifier, set())
            return RCTCATEResult(cate_fit, self._treatment, self._outcome, self._modifier, self._dag, len(data))

        result = smf.ols(f"{self._outcome} ~ {self._treatment}", data=data).fit()
        return RCTResult(result, self._treatment, self._outcome, self._dag, len(data))
