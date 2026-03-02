"""
Narrative explanation renderer for causal estimation results.

Each public ``explain_*`` function takes a fitted result object (which must
have ``self._dag`` set) and returns a formatted multi-line string.  The
result's ``executive_summary()`` method calls the appropriate function here.
"""
from __future__ import annotations

_SEP = "━" * 66


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.3f}"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def _list_vars(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _effect_phrase(effect: float, treatment: str, outcome: str) -> str:
    direction = "increase" if effect >= 0 else "decrease"
    return (
        f"a one-unit increase in {treatment} is estimated to cause "
        f"an {direction} of {abs(effect):.4f} in {outcome}"
    )


def _binary_effect_phrase(effect: float, treatment: str, outcome: str) -> str:
    direction = "increase" if effect >= 0 else "decrease"
    return (
        f"among treated units, receiving {treatment} is estimated to cause "
        f"an {direction} of {abs(effect):.4f} in {outcome} compared to not being treated"
    )


# ── Section builders ───────────────────────────────────────────────────────────

def _dag_section(dag, treatment: str, outcome: str, adjustment_set: set[str]) -> str:
    adj = sorted(adjustment_set)
    lines = [
        "CAUSAL STRUCTURE (DAG)",
        "The following directed causal relationships were assumed:",
    ]
    for cause, effect in dag.edges:
        lines.append(f"  • {cause} → {effect}")

    lines.append("")
    if adj:
        n = len(adj)
        var_str = _list_vars(adj)
        confounder_s = "a confounder" if n == 1 else "confounders"
        control_s = "a control variable" if n == 1 else "control variables"
        lines.append(
            f"The backdoor criterion identifies {var_str} as {confounder_s} "
            f"— {'a common cause' if n == 1 else 'common causes'} of both {treatment} and "
            f"{outcome} that must be held constant. "
            f"{'It is' if n == 1 else 'They are'} included as {control_s}."
        )
    else:
        lines.append(
            f"No confounders of {treatment} and {outcome} were identified in the DAG."
        )
    return "\n".join(lines)


def _assumptions_section(assumptions: list) -> str:
    n = len(assumptions)
    n_u = sum(1 for a in assumptions if not a.testable)
    n_t = n - n_u

    if n_u == n:
        intro = (
            f"All {n} required assumptions are untestable from the data alone "
            f"and must be justified on substantive grounds."
        )
    elif n_t == n:
        intro = f"All {n} required assumptions can be empirically checked in the data."
    else:
        intro = (
            f"{n_u} of the {n} required assumptions {'is' if n_u == 1 else 'are'} untestable "
            f"and must be justified on substantive grounds; "
            f"{n_t} {'can' if n_t == 1 else 'can'} be checked in the data."
        )

    lines = ["ASSUMPTIONS", intro, ""]
    for a in assumptions:
        tag = "[  testable  ]" if a.testable else "[ untestable ]"
        lines.append(f"  {tag}  {a.name}")
    return "\n".join(lines)


# ── Method-specific explanations ───────────────────────────────────────────────

def explain_ols(result) -> str:
    T, Y = result._treatment, result._outcome
    adj = sorted(result._adjustment_set)
    lo, hi = result.conf_int
    bias = result.unadjusted_effect - result.effect

    bias_lines = []
    if adj:
        bias_dir = "upward" if bias > 0 else "downward"
        bias_lines = [
            "",
            f"For comparison, the unadjusted (naive) estimate was "
            f"{result.unadjusted_effect:.4f}. The difference of {abs(bias):.4f} "
            f"reflects {bias_dir} confounding bias that is removed by "
            f"controlling for {_list_vars(adj)}.",
        ]

    blocks = [
        "\n".join([_SEP, f"Executive Summary — OLS Observational Regression",
                   f"  {T} → {Y}", _SEP]),

        "\n".join([
            "METHOD",
            f"Ordinary Least Squares (OLS) regression estimates the causal effect of "
            f"{T} on {Y} by adjusting for confounders identified from the DAG via the "
            f"backdoor criterion. This approach is valid when all relevant confounders "
            f"are observed and the relationship between variables is approximately linear.",
        ]),

        _dag_section(result._dag, T, Y, result._adjustment_set),
        _assumptions_section(result.assumptions),

        "\n".join([
            "RESULT",
            f"Controlling for {_list_vars(adj)}, {_effect_phrase(result.effect, T, Y)} "
            f"(95% CI: {_fmt_ci(lo, hi)}, SE = {result.std_err:.4f}, "
            f"{_fmt_p(result.pvalue)})." if adj else
            f"{_effect_phrase(result.effect, T, Y).capitalize()} "
            f"(95% CI: {_fmt_ci(lo, hi)}, SE = {result.std_err:.4f}, "
            f"{_fmt_p(result.pvalue)}).",
            *bias_lines,
        ]),

        "\n".join([
            "CAVEATS",
            f"This estimate is only as credible as the DAG. Confounders not declared "
            f"in the DAG cannot be detected or adjusted for, and their presence will "
            f"bias the result in an unknown direction. The estimate should not be "
            f"interpreted causally if any untestable assumption is likely violated.",
        ]),

        _SEP,
    ]
    return "\n\n".join(blocks)


def explain_iv(result) -> str:
    T, Y, Z = result._treatment, result._outcome, result._instrument
    adj = sorted(result._adjustment_set)
    lo, hi = result.conf_int
    bias = result.unadjusted_effect - result.effect

    controls_note = f", with {_list_vars(adj)} included as {'a control' if len(adj) == 1 else 'controls'}" if adj else ""

    bias_lines = [
        "",
        f"The unadjusted (naive) OLS estimate was {result.unadjusted_effect:.4f}. "
        f"The difference of {abs(bias):.4f} reflects the confounding bias that "
        f"IV corrects for by isolating only the variation in {T} driven by {Z}.",
    ]

    blocks = [
        "\n".join([_SEP, f"Executive Summary — Instrumental Variables (2SLS)",
                   f"  {T} → {Y}  |  instrument: {Z}", _SEP]),

        "\n".join([
            "METHOD",
            f"Two-Stage Least Squares (2SLS) uses {Z} as an instrument to isolate "
            f"exogenous variation in {T}, bypassing the need to observe all confounders. "
            f"In the first stage, {Z} predicts {T}. In the second stage, only the "
            f"variation in {T} attributable to {Z} is used to estimate the effect on {Y}. "
            f"The result is a Local Average Treatment Effect (LATE): the causal effect "
            f"for units whose treatment status is moved by the instrument (compliers).",
        ]),

        _dag_section(result._dag, T, Y, result._adjustment_set),
        _assumptions_section(result.assumptions),

        "\n".join([
            "RESULT",
            f"Among compliers (units whose {T} is affected by {Z}), "
            f"{_effect_phrase(result.effect, T, Y)}{controls_note} "
            f"(95% CI: {_fmt_ci(lo, hi)}, SE = {result.std_err:.4f}, "
            f"{_fmt_p(result.pvalue)}).",
            *bias_lines,
        ]),

        "\n".join([
            "CAVEATS",
            f"The exclusion restriction — that {Z} affects {Y} only through {T} — "
            f"is untestable and must be defended on substantive grounds. "
            f"Likewise, {Z} must be independent of unobserved confounders of {T} and {Y}. "
            f"If the instrument is weak (low first-stage F-statistic), 2SLS estimates "
            f"become unreliable and may be more biased than OLS. "
            f"The LATE applies only to compliers and may not generalise to the full population.",
        ]),

        _SEP,
    ]
    return "\n\n".join(blocks)


def explain_rct(result) -> str:
    T, Y = result._treatment, result._outcome
    lo, hi = result.conf_int

    blocks = [
        "\n".join([_SEP, f"Executive Summary — Randomized Controlled Trial",
                   f"  {T} → {Y}  |  estimand: ATE", _SEP]),

        "\n".join([
            "METHOD",
            f"Because {T} was randomly assigned, the Average Treatment Effect (ATE) "
            f"is estimated directly as the difference in mean {Y} between treated and "
            f"control units, equivalent to a simple OLS regression of {Y} on {T}. "
            f"Randomisation eliminates confounding, so no covariate adjustment is needed "
            f"or appropriate.",
        ]),

        _dag_section(result._dag, T, Y, set()),
        _assumptions_section(result.assumptions),

        "\n".join([
            "RESULT",
            f"{_effect_phrase(result.effect, T, Y).capitalize()} "
            f"(ATE = {result.effect:.4f}, 95% CI: {_fmt_ci(lo, hi)}, "
            f"SE = {result.std_err:.4f}, {_fmt_p(result.pvalue)}).",
        ]),

        "\n".join([
            "CAVEATS",
            f"This estimate is only valid if treatment was truly randomly assigned. "
            f"Non-compliance, attrition, or interference between units can invalidate "
            f"the causal interpretation even in a well-designed RCT. "
            f"The ATE is an average across all units and may mask heterogeneous effects.",
        ]),

        _SEP,
    ]
    return "\n\n".join(blocks)


def explain_matching(result) -> str:
    from .estimators.matching import _BOOTSTRAP_N
    T, Y = result._treatment, result._outcome
    adj = sorted(result._adjustment_set)
    lo, hi = result.conf_int
    bias = result.unadjusted_effect - result.effect

    blocks = [
        "\n".join([_SEP, f"Executive Summary — Propensity Score Matching",
                   f"  {T} → {Y}  |  estimand: ATT", _SEP]),

        "\n".join([
            "METHOD",
            f"Propensity Score Matching (PSM) estimates the Average Treatment Effect "
            f"on the Treated (ATT): the causal effect of {T} for units that actually "
            f"received it. Each treated unit is matched to its nearest control unit by "
            f"propensity score (estimated probability of treatment given {_list_vars(adj) if adj else 'covariates'}), "
            f"using 1-to-1 nearest-neighbour matching with replacement. "
            f"Standard errors and confidence intervals are computed via bootstrap "
            f"({_BOOTSTRAP_N} replicates).",
        ]),

        _dag_section(result._dag, T, Y, result._adjustment_set),
        _assumptions_section(result.assumptions),

        "\n".join([
            "RESULT",
            f"{_binary_effect_phrase(result.effect, T, Y).capitalize()} "
            f"(ATT = {result.effect:.4f}, 95% CI: {_fmt_ci(lo, hi)}, "
            f"SE = {result.std_err:.4f}, {_fmt_p(result.pvalue)}).",
            "",
            f"The unmatched (naive) mean difference was {result.unadjusted_effect:.4f}. "
            f"The difference of {abs(bias):.4f} is the estimated confounding bias "
            f"removed by matching on {_list_vars(adj) if adj else 'propensity score'}.",
        ]),

        "\n".join([
            "CAVEATS",
            f"Conditional independence — no unobserved confounders given the matched "
            f"variables — is untestable and is the central assumption of this approach. "
            f"Matching can only balance observed covariates; unobserved differences "
            f"between treated and control units that affect {Y} will still bias the ATT. "
            f"The ATT estimates the effect for those who were treated and may not "
            f"generalise to the untreated population.",
        ]),

        _SEP,
    ]
    return "\n\n".join(blocks)
