# TODO

## Relax DAG validation for policy-learning features

`RCTResult.learn_policy()` currently validates candidate features with
`_validate_modifier_dag` (shared with CATE effect modifiers), which requires
each feature to be an **ancestor of the outcome**. That requirement is
stricter than the statistics need:

- In an RCT, treatment is independent of every pre-treatment variable, so the
  doubly robust value estimate is honest for a policy conditioning on *any*
  pre-treatment feature — causal, proxy, or noise.
- Proxy features (e.g. `income → zip`, `income → outcome`: zip causes nothing
  but predicts who benefits) are valid and useful targeting features, yet are
  rejected when the DAG is specified honestly.
- Noise features can't inflate the honest out-of-fold value, and the
  placebo/random-modifier refutation checks already guard against them.

The **mediator exclusion must stay**: a descendant of the treatment is not
actionable at assignment time and conditioning on it creates selection bias.

Plan:

- Give policy learning its own validation: feature must be a DAG node and not
  a descendant of the treatment; drop the ancestor-of-outcome requirement.
- Keep the stricter check for CATE effect modifiers, where the output is an
  interpretive claim ("the effect differs by X") rather than just a rule and
  its value.
- Update the corresponding passage in `docs/causal/decisions.rst` (~line 231)
  and the `learn_policy` docstring in `formative/causal/estimators/rct.py`.
