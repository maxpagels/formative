# CLAUDE.md

## Commands

```bash
uv sync --dev                          # install dependencies
uv run pytest                          # run all tests
uv run pytest tests/test_iv.py         # run one file
uv run pytest tests/test_iv.py::TestIV2SLSEstimation::test_iv_recovers_true_effect  # run one test
uv run --group docs sphinx-build -b html docs docs/_build  # build docs
```

---

## Package structure

`formative` has two independent submodules:

```
formative/
  causal/    — causal effect estimation
  game/      — decision rules under uncertainty
```

---

## formative.causal

Three layers: DAG → estimators → refutations.

**DAG (`formative/causal/dag.py`)**
Built by calling `dag.assume(node).causes(*effects)`. Edges are validated as acyclic on insertion (Kahn's algorithm). Required first input to every estimator.

**Estimators (`formative/causal/estimators/`)**
Each takes a `DAG` + `treatment`/`outcome` in `__init__`, validates the DAG, then at `.fit(df)`:
- Runs `_identify()` using the backdoor criterion to find confounders (common ancestors of treatment and outcome, minus descendants of treatment).
- Raises `IdentificationError` if a DAG-declared confounder is absent from the data and cannot be handled. IV2SLS does not raise for unobserved confounders — the instrument handles them.
- Returns a result object with the main estimate and an unadjusted OLS estimate for comparison.

**Refutations (`formative/causal/refutations/`)**
Called as `result.refute(df)` — always pass the same `df` used for `fit()`. Returns a report containing `RefutationCheck` objects, each with `name`, `passed`, and `detail`. Checks are diagnostics, not proof; failing checks are signals to investigate.

**Refutation seeds** — isolated to avoid collisions with test data (seeded at 42):
- `_RCC_SEED = 54321` — random common cause column
- `_PLACEBO_SEED = 99999` — treatment permutation in matching
- `_BOOTSTRAP_SEED = 42` — safe to reuse in matching's bootstrap (samples row indices, not independent values)

Do not change `_RCC_SEED` or `_PLACEBO_SEED` to 42 — the generated noise collides with instrument/treatment data and produces a singular matrix in IV.

**Package layout:**
- `formative/causal/dag.py` — `DAG`, `_Node`
- `formative/causal/_exceptions.py` — `IdentificationError`, `GraphError`
- `formative/causal/estimators/ols.py` — `OLSObservational`, `OLSResult`
- `formative/causal/estimators/iv.py` — `IV2SLS`, `IVResult`
- `formative/causal/estimators/matching.py` — `PropensityScoreMatching`, `MatchingResult`; exports `_propensity_scores`, `_att_from_ps` for use by refutations
- `formative/causal/estimators/rct.py` — `RCT`, `RCTResult`
- `formative/causal/estimators/did.py` — `DiD`, `DiDResult`
- `formative/causal/estimators/rdd.py` — `RDD`, `RDDResult`
- `formative/causal/refutations/_check.py` — `Assumption`, `RefutationCheck`, `RefutationReport`
- `formative/causal/refutations/ols.py` — `OLSRefutationReport`, `_check_random_common_cause`
- `formative/causal/refutations/iv.py` — `IVRefutationReport`, `_check_first_stage_f`, `_check_random_common_cause`
- `formative/causal/refutations/matching.py` — `MatchingRefutationReport`, `_check_placebo_treatment`, `_check_random_common_cause`
- `formative/causal/refutations/rct.py` — `RCTRefutationReport`, `_check_random_common_cause`
- `formative/causal/refutations/did.py` — `DiDRefutationReport`, `_check_placebo_group`, `_check_random_common_cause`
- `formative/causal/refutations/rdd.py` — `RDDRefutationReport`, `_check_placebo_cutoff`, `_check_random_common_cause`
- `formative/causal/_explain.py` — narrative rendering for all estimators
- `formative/causal/decision.py` — `DecisionReport` (cost-benefit analysis on a causal estimate)
- `formative/causal/__init__.py` — public API

**Adding an estimator:** follow `OLSObservational`/`IV2SLS` — `__init__` validates DAG, `fit()` calls `_identify()`, raises `IdentificationError` where appropriate, returns a result object with a `refute()` method. Export from `__init__.py`. Add a refutations module under `formative/causal/refutations/`.

**Bootstrap-heavy tests:** matching tests use `setup_class` (not `setup_method`) so `.fit()` runs once per class. Use N=1_000 — larger N widens bootstrap SE faster than NN noise shrinks, causing refutation checks to fail even on well-specified data.

**Planned estimators:** the goal is to cover all methods in the causal inference decision tree at https://www.maxpagels.com/prototypes/causal-wizard. Remaining: **Synthetic Control**.

---

## formative.game

Decision rules for choosing between options under uncertainty. Each rule is a standalone function that accepts `{choice: {scenario: payoff}}` and returns a solver with a `.solve()` method.

```python
from formative.game import maximin

result = maximin({
    "stocks": {"recession": -20, "stagnation": 5, "growth": 30},
    "bonds":  {"recession":   5, "stagnation": 5, "growth":  7},
}).solve()
```

**Package layout:**
- `formative/game/maximin.py` — `maximin`, `Maximin`, `MaximinResult`
- `formative/game/maximax.py` — `maximax`, `Maximax`, `MaximaxResult`
- `formative/game/minimax_regret.py` — `minimax_regret`, `MinimaxRegret`, `MinimaxRegretResult`
- `formative/game/hurwicz.py` — `hurwicz`, `Hurwicz`, `HurwiczResult`
- `formative/game/laplace.py` — `laplace`, `Laplace`, `LaplaceResult`
- `formative/game/expected_value.py` — `expected_value`, `ExpectedValue`, `ExpectedValueResult`
- `formative/game/__init__.py` — public API

**Adding a decision rule** — every rule must have docs, tests, and follow these conventions:

- **Code:** a public function `rule_name(outcomes, ...)` returning a solver instance. The solver (`RuleName`) validates inputs in `__init__` and exposes `.solve()`. `.solve()` returns a `RuleNameResult` dataclass. Export all three from `__init__.py`.
- **Result dataclass:** fields are `choice: str`, then a scalar for the chosen option (e.g. `guaranteed`, `score`, `expected`), then any dicts. Include a custom `__repr__` that lists all choices with the key metric formatted as `{v:+.4g}` and the chosen marked `← chosen`.
- **Solver class docstring:** include an `Examples` block with `.solve()` and the correct `result.choice`.
- **Tests:** add `class TestRuleName` to `tests/test_game.py`. Cover: correct choice, scalar field, dict field, empty outcomes raises, `← chosen` in repr, and any rule-specific validation.
- **Docs:** add a section to `docs/game/decision_rules.rst` with a one-line question header, prose explanation, manual worked example, code block, expected output block, and `.. autoclass::` directives for solver and result.
