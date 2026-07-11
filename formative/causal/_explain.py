"""
Narrative explanation renderer for causal estimation results.

Each public ``explain_*`` function takes a fitted result object (which must
have ``self._dag`` set) and returns a formatted multi-line string.  The
result's ``executive_summary()`` method calls the appropriate function here.
"""

from __future__ import annotations

from datetime import datetime

_SEP = "━" * 66


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def _fmt_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d  %H:%M")


def _list_vars(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _effect_phrase(effect: float, treatment: str, outcome: str) -> str:
    direction = "increase" if effect >= 0 else "decrease"
    return f"a one-unit increase in {treatment} is estimated to cause an {direction} of {abs(effect):.4f} in {outcome}"


def _binary_effect_phrase(effect: float, treatment: str, outcome: str) -> str:
    direction = "increase" if effect >= 0 else "decrease"
    return (
        f"among treated units, receiving {treatment} is estimated to cause "
        f"an {direction} of {abs(effect):.4f} in {outcome} compared to not being treated"
    )


# ── Section builders ───────────────────────────────────────────────────────────


def _pluralize(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def _dag_section(dag, treatment: str, outcome: str, adjustment_set: set[str]) -> str:
    adj = sorted(adjustment_set)
    lines = [
        "CAUSAL STRUCTURE",
        "The following directed causal relationships were assumed:",
    ]
    for cause, effect in dag.edges:
        lines.append(f"  • {cause} → {effect}")

    lines.append("")
    if adj:
        n = len(adj)
        lines.append(
            f"The causal diagram identifies {_list_vars(adj)} as "
            f"{_pluralize(n, 'a variable', 'variables')} that influence{_pluralize(n, 's', '')} "
            f"both {treatment} and {outcome} and must be held constant to isolate the effect. "
            f"{_pluralize(n, 'It is', 'They are')} included as "
            f"{_pluralize(n, 'a control variable', 'control variables')}."
        )
    else:
        lines.append(
            f"No variables that influence both {treatment} and {outcome} were identified in the causal diagram."
        )
    return "\n".join(lines)


def _heterogeneity_section(result) -> str | None:
    """HETEROGENEITY block for results fitted with an effect modifier, else None."""
    modifier = getattr(result, "_modifier", None)
    if modifier is None:
        return None

    groups = result.effect_by_group
    p = result.homogeneity_pvalue
    largest = max(groups, key=lambda g: g.effect)
    smallest = min(groups, key=lambda g: g.effect)

    if p < 0.05:
        verdict = (
            f"A formal test rejects the hypothesis that the effect is the same across "
            f"{modifier} groups ({_fmt_p(p)}): the treatment effect genuinely differs by {modifier}."
        )
    else:
        verdict = (
            f"A formal test does not reject the hypothesis that the effect is the same "
            f"across {modifier} groups ({_fmt_p(p)}); the differences below may be noise."
        )

    lines = [
        "HETEROGENEITY",
        f"The effect was allowed to differ across levels of {modifier}. {verdict}",
        "",
    ]
    for g in groups:
        lo, hi = g.conf_int
        lines.append(f"  • {modifier} = {g.level}: effect = {g.effect:.4f}  (95% CI: {_fmt_ci(lo, hi)}, n = {g.n:,})")
    lines += [
        "",
        f"The largest estimated effect is for {modifier} = {largest.level} ({largest.effect:.4f}) "
        f"and the smallest for {modifier} = {smallest.level} ({smallest.effect:.4f}). Group "
        f"estimates are less precise than the overall average, and with many groups some "
        f"extreme-looking effects are expected by chance.",
    ]
    return "\n".join(lines)


def _assumptions_section(assumptions: list) -> str:
    n = len(assumptions)
    n_u = sum(1 for a in assumptions if not a.testable)
    n_t = n - n_u

    if n_u == n:
        intro = (
            f"All {n} required assumptions cannot be verified from the data alone "
            f"and must be justified based on subject-matter knowledge."
        )
    elif n_t == n:
        intro = f"All {n} required assumptions can be checked empirically in the data."
    else:
        intro = (
            f"{n_u} of the {n} required assumptions {_pluralize(n_u, 'cannot', 'cannot')} be "
            f"verified from the data alone and must be justified based on subject-matter "
            f"knowledge; {n_t} can be checked empirically."
        )

    lines = ["ASSUMPTIONS", intro, ""]
    for a in assumptions:
        lines.append(f"  {a.fmt_tag()}  {a.name}")
    return "\n".join(lines)


# ── Method-specific explanations ───────────────────────────────────────────────


def explain_ols(result) -> str:
    T, Y = result._treatment, result._outcome
    adj = sorted(result._adjustment_set)
    lo, hi = result.conf_int
    bias = result.unadjusted_effect - result.effect
    heterogeneity = _heterogeneity_section(result)

    bias_lines = []
    if adj:
        bias_dir = "upward" if bias > 0 else "downward"
        bias_lines = [
            "",
            f"For comparison, the unadjusted estimate (ignoring shared causes) was "
            f"{result.unadjusted_effect:.4f}. The difference of {abs(bias):.4f} "
            f"reflects {bias_dir} bias that is removed by "
            f"holding {_list_vars(adj)} constant.",
        ]

    blocks = [
        "\n".join(
            [
                _SEP,
                "Executive Summary — OLS Observational Regression",
                f"  {T} → {Y}",
                f"  Generated: {_fmt_timestamp()}",
                _SEP,
            ]
        ),
        "\n".join(
            [
                "METHOD",
                f"Standard regression estimates the causal effect of {T} on {Y} by "
                f"holding constant variables identified in the causal diagram that "
                f"influence both {T} and {Y}. This approach is valid when all such "
                f"variables are measured and the relationship between variables is "
                f"approximately linear.",
            ]
        ),
        _dag_section(result._dag, T, Y, result._adjustment_set),
        _assumptions_section(result.assumptions),
        "\n".join(
            [
                "RESULT",
                (
                    f"Holding {_list_vars(adj)} constant, {_effect_phrase(result.effect, T, Y)} "
                    if adj
                    else f"{_effect_phrase(result.effect, T, Y).capitalize()} "
                )
                + f"(95% CI: {_fmt_ci(lo, hi)}, SE = {result.std_err:.4f}, {_fmt_p(result.pvalue)}).",
                *bias_lines,
            ]
        ),
        *([heterogeneity] if heterogeneity else []),
        "\n".join(
            [
                "CAVEATS",
                f"This estimate is only as credible as the causal diagram. Variables that "
                f"influence both {T} and {Y} but are not included in the diagram cannot be "
                f"detected or controlled for, and their presence will bias the result in an "
                f"unknown direction. The estimate should not be interpreted as causal if "
                f"any untestable assumption is likely violated.",
            ]
        ),
        _SEP,
    ]
    return "\n\n".join(blocks)


def explain_iv(result) -> str:
    T, Y, Z = result._treatment, result._outcome, result._instrument
    adj = sorted(result._adjustment_set)
    lo, hi = result.conf_int
    bias = result.unadjusted_effect - result.effect

    controls_label = "a control" if len(adj) == 1 else "controls"
    controls_note = f", with {_list_vars(adj)} included as {controls_label}" if adj else ""

    bias_lines = [
        "",
        f"The unadjusted estimate (ignoring hidden common causes) was "
        f"{result.unadjusted_effect:.4f}. The difference of {abs(bias):.4f} reflects "
        f"the bias that the instrumental variable approach corrects for by using only "
        f"the variation in {T} driven by {Z}.",
    ]

    blocks = [
        "\n".join(
            [
                _SEP,
                "Executive Summary — Instrumental Variables (2SLS)",
                f"  {T} → {Y}  |  instrument: {Z}",
                f"  Generated: {_fmt_timestamp()}",
                _SEP,
            ]
        ),
        "\n".join(
            [
                "METHOD",
                f"Instrumental Variables (IV) uses {Z} as a lever that shifts {T} without "
                f"directly affecting {Y}. By isolating only the part of {T}'s variation "
                f"driven by {Z}, the method sidesteps hidden common causes that would "
                f"otherwise distort a simple comparison. In the first stage, {Z} predicts "
                f"{T}. In the second stage, only that predicted variation is used to "
                f"estimate the effect on {Y}. The result applies specifically to units "
                f"whose treatment status was changed by {Z}.",
            ]
        ),
        _dag_section(result._dag, T, Y, result._adjustment_set),
        _assumptions_section(result.assumptions),
        "\n".join(
            [
                "RESULT",
                f"Among units whose {T} was shifted by {Z}, "
                f"{_effect_phrase(result.effect, T, Y)}{controls_note} "
                f"(95% CI: {_fmt_ci(lo, hi)}, SE = {result.std_err:.4f}, "
                f"{_fmt_p(result.pvalue)}).",
                *bias_lines,
            ]
        ),
        "\n".join(
            [
                "CAVEATS",
                f"The key assumption — that {Z} affects {Y} only through {T} and not "
                f"through any other path — cannot be tested and must be defended on "
                f"substantive grounds. {Z} must also be unrelated to any hidden factors "
                f"that affect both {T} and {Y}. If {Z} is only weakly related to {T}, "
                f"the IV estimate can be unreliable. The estimated effect applies only "
                f"to units whose treatment was moved by {Z} and may not reflect the "
                f"effect across the full population.",
            ]
        ),
        _SEP,
    ]
    return "\n\n".join(blocks)


def explain_rct(result) -> str:
    T, Y = result._treatment, result._outcome
    lo, hi = result.conf_int
    heterogeneity = _heterogeneity_section(result)

    blocks = [
        "\n".join(
            [
                _SEP,
                "Executive Summary — Randomized Controlled Trial",
                f"  {T} → {Y}  |  estimand: average effect across all units",
                f"  Generated: {_fmt_timestamp()}",
                _SEP,
            ]
        ),
        "\n".join(
            [
                "METHOD",
                f"Because {T} was randomly assigned, the average causal effect is estimated "
                f"directly as the difference in mean {Y} between treated and control units. "
                f"Random assignment means treated and control groups are comparable on "
                f"everything except treatment, so no additional adjustments are needed.",
            ]
        ),
        _dag_section(result._dag, T, Y, set()),
        _assumptions_section(result.assumptions),
        "\n".join(
            [
                "RESULT",
                f"{_effect_phrase(result.effect, T, Y).capitalize()} "
                f"(average effect = {result.effect:.4f}, 95% CI: {_fmt_ci(lo, hi)}, "
                f"SE = {result.std_err:.4f}, {_fmt_p(result.pvalue)}).",
            ]
        ),
        *([heterogeneity] if heterogeneity else []),
        "\n".join(
            [
                "CAVEATS",
                "This estimate is only valid if treatment was truly randomly assigned. "
                "Non-compliance (some units not following their assigned treatment), "
                "drop-out, or units influencing each other can undermine the causal "
                "interpretation even in a well-designed trial. The estimate is an average "
                "across all units and may mask variation in how different groups respond.",
            ]
        ),
        _SEP,
    ]
    return "\n\n".join(blocks)


def explain_did(result) -> str:
    G, T, Y = result._group, result._time, result._outcome
    lo, hi = result.conf_int
    baseline_bias = result.naive_diff - result.effect

    blocks = [
        "\n".join(
            [
                _SEP,
                "Executive Summary — Difference-in-Differences",
                f"  ({G} \u00d7 {T}) \u2192 {Y}  |  estimand: effect on the treated group",
                f"  Generated: {_fmt_timestamp()}",
                _SEP,
            ]
        ),
        "\n".join(
            [
                "METHOD",
                f"Difference-in-Differences estimates the effect of treatment on the "
                f"treated group by comparing how {Y} changed over time for the treated "
                f"group ({G} = 1) versus the control group ({G} = 0). Any trend that "
                f"affected both groups equally cancels out, leaving only the treatment "
                f"effect. The estimate is the interaction term in a regression of {Y} "
                f"on group, time period, and their combination ({G}:{T}).",
            ]
        ),
        "\n".join(
            [
                "CAUSAL STRUCTURE",
                "The following directed causal relationships were assumed:",
                *[f"  \u2022 {cause} \u2192 {effect}" for cause, effect in result._dag.edges],
                "",
                "Identification here comes from the panel design — specifically the "
                "assumption that both groups would have followed the same trend without "
                "treatment — rather than from controlling for observed variables.",
            ]
        ),
        _assumptions_section(result.assumptions),
        "\n".join(
            [
                "RESULT",
                f"The estimated treatment effect is {result.effect:.4f} "
                f"(95% CI: {_fmt_ci(lo, hi)}, SE = {result.std_err:.4f}, {_fmt_p(result.pvalue)}). "
                f"The simple post-period difference (treated minus control) was {result.naive_diff:.4f}; "
                f"the method removes {abs(baseline_bias):.4f} of pre-existing trend "
                f"({'upward' if baseline_bias > 0 else 'downward'} bias in the simple comparison).",
            ]
        ),
        "\n".join(
            [
                "CAVEATS",
                "The central assumption — that treated and control groups would have "
                "followed the same trajectory without treatment — cannot be tested "
                "directly. If the groups were already on different paths before treatment, "
                "the estimate will be biased even with panel data. Pre-treatment trend "
                "checks (where multiple pre-periods are available) can offer partial "
                "reassurance but cannot confirm the assumption.",
            ]
        ),
        _SEP,
    ]
    return "\n\n".join(blocks)


def explain_rdd(result) -> str:
    R, T, Y = result._running_var, result._treatment, result._outcome
    lo, hi = result.conf_int
    bw_label = f"{result.bandwidth:.4f}" if result.bandwidth is not None else "all observations"
    bias = result.unadjusted_effect - result.effect
    bias_dir = "upward" if bias > 0 else "downward"

    blocks = [
        "\n".join(
            [
                _SEP,
                "Executive Summary — Regression Discontinuity Design",
                f"  {R} \u2192 {T} \u2192 {Y}  |  cutoff: {result.cutoff}",
                f"  Generated: {_fmt_timestamp()}",
                _SEP,
            ]
        ),
        "\n".join(
            [
                "METHOD",
                f"Regression Discontinuity Design exploits the fact that treatment ({T}) "
                f"is assigned sharply when {R} crosses {result.cutoff}. Units just above "
                f"and just below the threshold are locally comparable — they differ "
                f"essentially at random with respect to the threshold rule. The jump in "
                f"{Y} at the cutoff therefore identifies the causal effect of {T} for "
                f"units near the threshold. Identification is achieved by fitting a local "
                f"linear regression that allows different slopes on each side of the cutoff "
                f"and reading off the discontinuous jump as the treatment effect.",
            ]
        ),
        "\n".join(
            [
                "CAUSAL STRUCTURE",
                "The following directed causal relationships were assumed:",
                *[f"  \u2022 {cause} \u2192 {effect}" for cause, effect in result._dag.edges],
                "",
                f"Identification comes from the threshold rule — {R} crossing {result.cutoff} "
                f"determines {T} — rather than from controlling for observed confounders. "
                f"The bandwidth ({bw_label}) restricts estimation to observations close "
                f"enough to the cutoff for the local comparability assumption to hold.",
            ]
        ),
        _assumptions_section(result.assumptions),
        "\n".join(
            [
                "RESULT",
                f"The estimated LATE at the cutoff is {result.effect:.4f} "
                f"(95% CI: {_fmt_ci(lo, hi)}, SE = {result.std_err:.4f}, "
                f"{_fmt_p(result.pvalue)}), based on {result.n_obs:,} observations. "
                f"The naive mean difference (above minus below cutoff) was "
                f"{result.unadjusted_effect:.4f}; the local linear regression removes "
                f"{abs(bias):.4f} of {bias_dir} bias introduced by the slope of {R} "
                f"on {Y}.",
            ]
        ),
        "\n".join(
            [
                "CAVEATS",
                "The LATE applies only to units near the cutoff and may not generalise "
                "to the wider population. The continuity assumption — that potential "
                "outcomes change smoothly at the cutoff — cannot be tested; if units "
                "can manipulate which side of the threshold they fall on, the local "
                "comparability breaks down. The estimate is also sensitive to the choice "
                "of bandwidth and functional form (linear vs higher-order polynomial) "
                "used for the local regression.",
            ]
        ),
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
        "\n".join(
            [
                _SEP,
                "Executive Summary — Propensity Score Matching",
                f"  {T} → {Y}  |  estimand: effect on those who received treatment",
                f"  Generated: {_fmt_timestamp()}",
                _SEP,
            ]
        ),
        "\n".join(
            [
                "METHOD",
                f"Propensity Score Matching estimates the effect of {T} specifically for "
                f"units that received it. Each treated unit is paired with the most similar "
                f"untreated unit based on their estimated likelihood of receiving treatment "
                f"given {_list_vars(adj) if adj else 'the available covariates'}. "
                f"This creates a comparison group that resembles the treated group on "
                f"observed characteristics. Standard errors are computed by repeating the "
                f"matching procedure on {_BOOTSTRAP_N} resampled datasets.",
            ]
        ),
        _dag_section(result._dag, T, Y, result._adjustment_set),
        _assumptions_section(result.assumptions),
        "\n".join(
            [
                "RESULT",
                f"{_binary_effect_phrase(result.effect, T, Y).capitalize()} "
                f"(estimated effect = {result.effect:.4f}, 95% CI: {_fmt_ci(lo, hi)}, "
                f"SE = {result.std_err:.4f}, {_fmt_p(result.pvalue)}).",
                "",
                f"The raw unmatched mean difference was {result.unadjusted_effect:.4f}. "
                f"The difference of {abs(bias):.4f} is the estimated bias removed "
                f"by matching on {_list_vars(adj) if adj else 'the propensity score'}.",
            ]
        ),
        "\n".join(
            [
                "CAVEATS",
                f"Matching can only balance variables that are observed and included. "
                f"If there are unobserved differences between treated and untreated units "
                f"that affect {Y}, those will still bias the estimate. The result reflects "
                f"the effect for those who were treated and may not apply to the untreated "
                f"population.",
            ]
        ),
        _SEP,
    ]
    return "\n\n".join(blocks)


def explain_policy(result) -> str:
    T, Y = result._treatment, result._outcome
    lo, hi = result.value_ci
    mods = _list_vars(result.modifiers)

    if result.value > 0 and lo > 0:
        verdict = (
            f"The learned policy is estimated to add {result.value:.4f} in net benefit per "
            f"unit over the best one-size-fits-all option, and the confidence interval "
            f"excludes zero: the targeting appears genuinely valuable."
        )
    elif result.value > 0:
        verdict = (
            f"The learned policy is estimated to add {result.value:.4f} in net benefit per "
            f"unit over the best one-size-fits-all option, but the confidence interval "
            f"includes zero — the targeting may not beat simply treating everyone or no one."
        )
    else:
        verdict = (
            f"The learned policy is not estimated to beat the best one-size-fits-all "
            f"option (value = {result.value:.4f} per unit). Treat everyone or no one "
            f"according to decide() rather than targeting on {mods}."
        )

    rule_lines = result.rules.split("\n")

    blocks = [
        "\n".join(
            [
                _SEP,
                "Executive Summary — Learned Treatment Policy",
                f"  {T} → {Y}  |  estimand: policy value vs best constant policy",
                f"  Generated: {_fmt_timestamp()}",
                _SEP,
            ]
        ),
        "\n".join(
            [
                "METHOD",
                f"Rather than deciding whether to apply {T} to everyone, this analysis asks "
                f"who should receive it. Each unit's individual net benefit was estimated with "
                f"cross-fitted doubly robust scoring, and an exhaustive search found the "
                f"shallow decision rule over {mods} that maximises total net benefit. The "
                f"policy's value is estimated honestly: every unit is evaluated under a rule "
                f"learned without its part of the data, and compared against the best "
                f"constant policy (treat everyone or treat no one).",
            ]
        ),
        _dag_section(result._dag, T, Y, set()),
        _assumptions_section(result.assumptions),
        "\n".join(
            [
                "POLICY",
                *[f"  • {line}" for line in rule_lines],
                "",
                f"The rule treats {result.coverage:.1%} of units. {verdict} "
                f"(value = {result.value:.4f} per unit, 95% CI: {_fmt_ci(lo, hi)}, "
                f"SE = {result.value_se:.4f}).",
            ]
        ),
        "\n".join(
            [
                "CAVEATS",
                "The value estimate describes the learning procedure, evaluated out-of-fold; "
                "the printed rule was re-learned on the full sample and may differ slightly "
                "from the fold-level rules behind the estimate. The tree will use its full "
                "depth budget even when the data support a simpler rule — compare the value "
                "of a depth-1 and depth-2 policy before shipping the more complex one, and "
                "run refute() to check that the value survives placebo features.",
            ]
        ),
        _SEP,
    ]
    return "\n\n".join(blocks)
