# Design note: policy learning (`learn_policy()`)

**Status:** implemented (see `formative/causal/estimators/policy.py`); deviations from this
note: refutation checks test the honest value rather than tree structure (in-sample search
splits on noise within its depth budget, so structural checks are flaky), and final trees get
a simplification pass that drops splits which don't change the assignment.
**Date:** 2026-07-11

## Motivation

formative's decision layer currently answers two questions:

- `result.decide(cost=, benefit=)` — *should I treat everyone?* Converts an average
  effect into a treat/don't-treat call via cost-benefit analysis.
- `result.decide_by_group()` — *should I treat each segment?* Same call per level of a
  single, user-chosen `effect_modifier`.

Both require the user to pick the segmentation up front. The next step is learning the
assignment policy itself: given several candidate pre-treatment features, find the
treatment rule that maximises net benefit, and report an honest estimate of that
policy's value. The output should be a shallow decision tree — "treat if
`tenure_band = <2y` and `region = North`" — because a rule a stakeholder can read,
audit, and ship is worth more than an opaque score.

This is the logical endpoint of the package's decision-first philosophy, and no
mainstream Python library makes it ergonomic: EconML buries it in
`SingleTreePolicyInterpreter`, and `policytree` is R-only.

## Prior art

- Athey & Wager (2021), *Policy Learning With Observational Data* — doubly robust
  scoring plus empirical welfare maximisation over a restricted policy class, with
  regret guarantees.
- Zhou, Athey & Wager (2023) and the R `policytree` package — exact shallow-tree
  search over doubly robust scores (C++ backend).
- Kitagawa & Tetsuya (2018) — empirical welfare maximisation framing.

## Proposed API

```python
result = RCT(dag, treatment="training", outcome="earnings").fit(df)

policy = result.learn_policy(
    df,                                   # same df passed to fit(), like refute()
    modifiers=["tenure_band", "region"],  # candidate discrete pre-treatment columns
    cost=2.5,                             # cost of treating one unit
    benefit=1.0,                          # value of one unit of outcome
    max_depth=2,                          # tree depth; 1 or 2
)

print(policy.summary())
```

`learn_policy()` returns a `PolicyResult` with:

- `rules` — the learned tree, both as a structured object and as readable text
  ("treat if tenure_band = <2y and region = North").
- `value` / `value_se` / `value_ci` — doubly robust estimate of the policy's net
  benefit per unit, reported as the **advantage over the best constant policy**
  (treat-all or treat-none, whichever is better), so a useless policy reads as ≈ 0.
- `coverage` — share of the sample the policy treats.
- `assign(df)` — apply the rule to new data, returning a boolean Series.
- `summary()` / `explain()` — consistent with the other result objects.

## Method

1. **Doubly robust scores.** For each unit, compute the AIPW score
   `Γᵢ = μ̂₁(xᵢ) − μ̂₀(xᵢ) + Wᵢ/ê · (Yᵢ − μ̂₁(xᵢ)) − (1−Wᵢ)/(1−ê) · (Yᵢ − μ̂₀(xᵢ))`,
   with outcome models `μ̂₀`, `μ̂₁` fit by OLS on the modifiers and cross-fitted
   (K = 5) to avoid own-observation overfitting. In the RCT case the propensity `ê`
   is the treated share (or known design probability), which is what makes RCT the
   right v1 scope — no propensity model to get wrong.
2. **Net-benefit transform.** `γᵢ = benefit · Γᵢ − cost` is the net gain of treating
   unit *i*. A policy's estimated value is the mean of `γᵢ` over the units it treats.
3. **Exact tree search.** Enumerate all depth ≤ `max_depth` trees whose splits are
   levels of the candidate discrete modifiers, and pick the one maximising total net
   benefit. With discrete modifiers this is a small exhaustive search in pure numpy —
   no C++ needed, which is why v1 does not support continuous splits.
4. **Honest value estimate.** In-sample value of the *chosen* policy is optimistic
   (winner's curse). Estimate value on held-out folds: learn the tree on K−1 folds,
   score on the held-out fold, aggregate. SE via the influence-function variance of
   the fold-level estimates.

## Scope decisions

- **v1 estimator: `RCTResult` only.** Known propensity makes the DR scores simple and
  trustworthy. `OLSObservational` support can follow, reusing
  `matching._propensity_scores` for `ê`; IV/DiD/RDD policy learning is out of scope
  (different estimands).
- **Discrete modifiers only**, matching the existing `effect_modifier` convention.
  Continuous features must be binned by the user. Keeps exact search tractable and the
  output rules readable.
- **Modifier validation reuses `_cate.py`:** reject columns that are descendants of
  the treatment (mediators) or not ancestors of the outcome, per the DAG.
- **`max_depth ≤ 2`.** Deeper trees stop being auditable, and search cost grows
  combinatorially. Enforced, not just defaulted.
- **No new dependencies.** OLS and (later) logistic regression via statsmodels;
  search in numpy.

## Refutations

`PolicyResult.refute(df)` following the existing pattern:

- **Placebo modifiers** — permute the modifier columns (seed following the
  `_PLACEBO_MODIFIER_SEED` convention; pick a fresh seed, see the collision warning in
  CLAUDE.md) and re-learn. The learned policy's advantage over the best constant
  policy should collapse to ≈ 0.
- **Random modifier** — add a pure-noise column to the candidates; the learned tree
  should not split on it, and value should not improve.

## Conventions checklist (from CLAUDE.md)

- Export `PolicyResult` from `formative/causal/__init__.py`.
- Refutations module at `formative/causal/refutations/policy.py`.
- Narrative rendering in `_explain.py`.
- Tests in `tests/test_policy.py` + `tests/test_policy_refutation.py`; cross-fitting
  makes fitting expensive, so use `setup_class` like the matching tests.
- Docs page under `docs/causal/` with a worked example; consider a benchmark notebook
  once a public dataset with a known-good policy exists.

## Open questions

- Should `value` also be reported against treat-none specifically (absolute net
  benefit), in addition to the best-constant-policy advantage?
- Tie-breaking when several trees have statistically indistinguishable value — prefer
  shallower/simpler?
- Multi-valued (non-binary) treatments — out of scope for v1, but the AIPW scoring
  generalises; worth keeping the internal score representation per-arm?
