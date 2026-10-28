from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .._assumptions import Assumption
from ..dag import DAG
from ._base import _backdoor_confounders, _BaseResult, _missing_confounders_error, _reject_missing

DML_ASSUMPTIONS: list[Assumption] = [
    Assumption("No unobserved confounders (selection on observables)", testable=False),
    Assumption("Overlap: every unit could have received any treatment level", testable=True),
    Assumption("Nuisance models predict outcome and treatment well enough", testable=True),
    Assumption("Stable Unit Treatment Value Assumption (SUTVA)", testable=False),
]

_DML_FOLD_SEED = 24680


# ── Private helpers (also imported by formative/refutations/dml.py) ───────────


def _is_binary(series: pd.Series) -> bool:
    return set(series.unique()) <= {0, 1}


def _fit_dml(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: set[str],
    ml_outcome,
    ml_treatment,
    n_folds: int,
):
    """
    Fit a DoubleML model with fixed fold assignment and return it.

    Binary treatment → ``DoubleMLIRM`` (fully non-parametric ATE, classifier
    propensity model). Otherwise → ``DoubleMLPLR`` (partially linear).
    Learners are cloned so repeated fits (refutations) never share state.
    """
    try:
        import doubleml as dml
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import KFold
    except ImportError as e:
        raise ImportError("DoubleML requires the 'dml' extra: pip install 'formative-ds[dml]'") from e

    binary = _is_binary(data[treatment])
    controls = sorted(adjustment_set)
    if ml_outcome is None:
        ml_outcome = RandomForestRegressor(min_samples_leaf=5, random_state=_DML_FOLD_SEED)
    if ml_treatment is None and binary:
        # Larger leaves keep propensities off 0/1, where IRM's inverse weights explode;
        # all features per split (the regressor default) so noise columns can't crowd out confounders.
        ml_treatment = RandomForestClassifier(min_samples_leaf=20, max_features=None, random_state=_DML_FOLD_SEED)
    elif ml_treatment is None:
        ml_treatment = RandomForestRegressor(min_samples_leaf=5, random_state=_DML_FOLD_SEED)

    # One-hot text/category confounders: the ML learners only take numbers.
    X = pd.get_dummies(data[controls], dtype=float)
    dml_data = dml.DoubleMLData(
        pd.concat([data[[outcome, treatment]], X], axis=1), y_col=outcome, d_cols=treatment, x_cols=list(X.columns)
    )
    if binary:
        model = dml.DoubleMLIRM(
            dml_data, ml_g=clone(ml_outcome), ml_m=clone(ml_treatment), n_folds=n_folds, draw_sample_splitting=False
        )
    else:
        model = dml.DoubleMLPLR(
            dml_data, ml_l=clone(ml_outcome), ml_m=clone(ml_treatment), n_folds=n_folds, draw_sample_splitting=False
        )
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=_DML_FOLD_SEED).split(np.zeros(len(data)))
    model.set_sample_splitting([list(folds)])
    return model.fit()


# ── Result ─────────────────────────────────────────────────────────────────────


class DoubleMLResult(_BaseResult):
    """
    The result of a Double Machine Learning estimation.

    Holds the debiased ATE alongside the unadjusted estimate (``outcome ~
    treatment`` only), so you can see the confounding bias directly. The
    fitted ``doubleml`` model is available as ``doubleml_model`` for
    sensitivity analysis, nuisance-model diagnostics, and ``gate()``/``cate()``.
    """

    _ASSUMPTIONS = DML_ASSUMPTIONS

    def __init__(
        self,
        model,
        unadjusted_result,
        treatment: str,
        outcome: str,
        adjustment_set: set[str],
        learners: tuple,
        n_folds: int,
        dag,
        n: int,
    ) -> None:
        self._model = model
        self._unadjusted = unadjusted_result
        self._treatment = treatment
        self._outcome = outcome
        self._adjustment_set = adjustment_set
        self._learners = learners
        self._n_folds = n_folds
        self._dag = dag
        self._n = n

    @property
    def effect(self) -> float:
        """Point estimate of the average treatment effect."""
        return float(self._model.coef[0])

    @property
    def std_err(self) -> float:
        """Standard error of the effect, from the orthogonal score."""
        return float(self._model.se[0])

    @property
    def conf_int(self) -> tuple[float, float]:
        """95% confidence interval for the effect."""
        ci = self._model.confint()
        return (float(ci.iloc[0, 0]), float(ci.iloc[0, 1]))

    @property
    def pvalue(self) -> float:
        """p-value for the effect (``H0: effect = 0``)."""
        return float(self._model.pval[0])

    @property
    def unadjusted_effect(self) -> float:
        """Unadjusted point estimate: naive regression without controlling for confounders."""
        return float(self._unadjusted.params[self._treatment])

    @property
    def adjustment_set(self) -> set[str]:
        """Variables controlled for to satisfy the backdoor criterion."""
        return self._adjustment_set

    @property
    def doubleml_model(self):
        """The underlying fitted ``doubleml`` model (``DoubleMLPLR`` or ``DoubleMLIRM``)."""
        return self._model

    @property
    def binary_treatment(self) -> bool:
        """``True`` if the treatment is 0/1 and the interactive (IRM) model was used."""
        return type(self._model).__name__ == "DoubleMLIRM"

    def executive_summary(self) -> str:
        """Narrative explanation of the method, DAG, assumptions, and result."""
        from .._explain import explain_dml

        return explain_dml(self)

    def summary(self) -> str:
        """Concise tabular summary of the ATE estimate, confidence interval, and assumptions."""
        lo, hi = self.conf_int
        model = type(self._model).__name__.removeprefix("DoubleML")
        adj = ", ".join(sorted(self._adjustment_set))
        lines = [
            "",
            f"Double ML Causal Effect ({model}): {self._treatment} → {self._outcome}",
            "─" * 50,
            f"  Adjusted estimate    : {self.effect:>10.4f}  (controlling for: {adj})",
            f"  Unadjusted estimate  : {self.unadjusted_effect:>10.4f}  (no controls)",
            f"  Confounding bias     : {self.unadjusted_effect - self.effect:>+10.4f}",
            "",
            f"  Std. error           : {self.std_err:>10.4f}",
            f"  95% CI               : [{lo:.4f}, {hi:.4f}]",
            f"  p-value              : {self.pvalue:>10.4f}",
            f"  N                    : {self._n:>10}",
            f"  Cross-fitting folds  : {self._n_folds:>10}",
        ]
        lines += self._assumptions_lines()
        return "\n".join(lines)

    def refute(self, data: pd.DataFrame):
        """
        Run refutation checks against this Double ML estimation.

        Currently runs:

        - **Random common cause**: adds a random noise column as an extra
          control and checks that the estimate does not shift by more than
          one standard error.
        - **Placebo treatment**: randomly permutes the treatment column; the
          placebo estimate's own 99% CI should cover zero.
        - **Nuisance model fit**: the outcome model's out-of-fold R² must be
          positive, and the treatment model's must not exceed 0.95.
        - **Overlap** (binary treatment only): at most 10% of units may have
          a propensity score outside [0.05, 0.95].

        Parameters
        ----------
        data : pd.DataFrame
            The same dataframe passed to ``fit()``.
        """
        from ..refutations.dml import (
            DoubleMLRefutationReport,
            _check_nuisance_fit,
            _check_overlap,
            _check_placebo_treatment,
            _check_random_common_cause,
        )

        args = (data, self._treatment, self._outcome, self._adjustment_set, *self._learners, self._n_folds)
        checks = [
            _check_random_common_cause(*args, self.effect, self.std_err),
            _check_placebo_treatment(*args),
            _check_nuisance_fit(self._model, data, self._treatment, self._outcome),
        ]
        if self.binary_treatment:
            checks.append(_check_overlap(self._model))
        return DoubleMLRefutationReport(checks=checks, treatment=self._treatment, outcome=self._outcome)


class DoubleML:
    """
    Double Machine Learning estimator with DAG-based confounder identification.

    Like ``OLSObservational``, but the confounders' influence on the treatment
    and the outcome is learned with flexible machine-learning models, so
    non-linear and interacting confounding is handled. Both the outcome and
    the treatment are residualised on the confounders with cross-fitting, and
    the effect is estimated from the residuals — giving a valid standard error
    despite the ML nuisance models (Chernozhukov et al., 2018).

    - **Binary treatment** (0/1): interactive regression model (AIPW). The
      outcome model is fully non-parametric, so the ATE is valid under any
      effect heterogeneity. ``ml_treatment`` must be a classifier.
    - **Continuous treatment**: partially linear model. The estimate is an
      average marginal effect; a dose effect that is non-linear in the
      treatment is summarised, not traced out.

    Requires the ``dml`` extra: ``pip install 'formative-ds[dml]'``.

    Example::

        dag = DAG()
        dag.assume("ability").causes("education", "income")
        dag.assume("education").causes("income")

        result = DoubleML(dag, treatment="education", outcome="income").fit(df)
        print(result.summary())

    Parameters
    ----------
    dag : DAG
        Causal assumptions; confounders are found with the backdoor criterion.
    treatment, outcome : str
        Column / node names.
    ml_outcome : sklearn regressor, optional
        Model for the outcome given confounders. Default: random forest.
    ml_treatment : sklearn estimator, optional
        Model for the treatment given confounders — a classifier when the
        treatment is binary, a regressor otherwise. Default: random forest.
    n_folds : int
        Cross-fitting folds.
    """

    def __init__(
        self,
        dag: DAG,
        treatment: str,
        outcome: str,
        ml_outcome=None,
        ml_treatment=None,
        n_folds: int = 5,
    ) -> None:
        self._dag = dag
        self._treatment = treatment
        self._outcome = outcome
        self._ml_outcome = ml_outcome
        self._ml_treatment = ml_treatment
        self._n_folds = n_folds
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        nodes = self._dag.nodes
        for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
            if var not in nodes:
                raise ValueError(f"{label} '{var}' is not a node in the DAG. Known nodes: {sorted(nodes)}")
        if self._treatment == self._outcome:
            raise ValueError("Treatment and outcome must be different variables.")
        if self._n_folds < 2:
            raise ValueError("n_folds must be at least 2.")

    def _identify(self, data_columns: set[str]) -> tuple[set[str], set[str]]:
        """Backdoor adjustment set, split into ``(observed, dag_confounders_not_in_data)``."""
        confounders = _backdoor_confounders(self._dag, self._treatment, self._outcome)
        return {c for c in confounders if c in data_columns}, {c for c in confounders if c not in data_columns}

    def fit(self, data: pd.DataFrame) -> DoubleMLResult:
        """
        Identify the adjustment set, then estimate the effect with Double ML.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain treatment, outcome, and every DAG confounder.

        Returns
        -------
        DoubleMLResult

        Raises
        ------
        ``IdentificationError``
            If confounders declared in the DAG are absent from the dataframe.
        ``ValueError``
            If treatment or outcome columns are missing, contain NaNs, or the
            DAG has no confounders (use ``OLSObservational`` or ``RCT`` then —
            there is nothing for the ML models to learn).
        """
        data_columns = set(data.columns)

        for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
            if var not in data_columns:
                raise ValueError(f"{label} column '{var}' not found in dataframe.")

        adjustment_set, dag_confounders_not_in_data = self._identify(data_columns)

        if dag_confounders_not_in_data:
            raise _missing_confounders_error(dag_confounders_not_in_data, self._treatment, self._outcome)
        if not adjustment_set:
            raise ValueError(
                f"The DAG has no confounders of '{self._treatment}' and '{self._outcome}'. "
                "Double ML has nothing to adjust for — use OLSObservational or RCT instead."
            )

        _reject_missing(data, [self._treatment, self._outcome, *sorted(adjustment_set)])
        unadjusted_result = smf.ols(f"{self._outcome} ~ {self._treatment}", data=data).fit()
        learners = (self._ml_outcome, self._ml_treatment)
        model = _fit_dml(data, self._treatment, self._outcome, adjustment_set, *learners, self._n_folds)

        return DoubleMLResult(
            model,
            unadjusted_result,
            self._treatment,
            self._outcome,
            adjustment_set,
            learners,
            self._n_folds,
            self._dag,
            len(data),
        )
