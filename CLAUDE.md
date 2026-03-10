# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies and the package in editable mode
uv sync --dev

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_iv.py
uv run pytest tests/test_iv.py::TestIV2SLSEstimation::test_iv_recovers_true_effect

# Build docs
make build-docs
# or directly:
uv run --group docs sphinx-build -b html docs docs/_build
```

## Package structure

`formative` has two independent submodules:

```
formative/
  causal/      — causal effect estimation
  game/        — decision theory and game theory
```

Public imports:
```python
from formative.causal import DAG, OLSObservational, IV2SLS, ...
from formative.game import maximin
```

---

## formative.causal

Built around three layers: DAG, estimators, and refutations.

**Layer 1 — DAG (`formative/causal/dag.py`)**

User builds a `DAG` by calling `dag.assume(node).causes(*effects)`. Edges are validated as acyclic on insertion (Kahn's algorithm). The DAG is the mandatory first input to every estimator.

**Layer 2 — Estimators (`formative/causal/estimators/`)**

Each estimator takes a `DAG` + `treatment`/`outcome` (and optionally an `instrument`) in `__init__`, validates the DAG structure, then at `.fit(df)` time:
- Runs `_identify()` using the backdoor criterion (confounders = common ancestors of treatment and outcome, minus descendants of treatment) to find which variables to control for.
- Raises `IdentificationError` if a DAG-declared confounder is absent from the data and cannot be handled (OLS); IV2SLS does not raise for unobserved confounders since the instrument handles them.
- Fits the model and returns a result object holding both the main estimate and a plain unadjusted OLS estimate for comparison.

**Layer 3 — Refutations (`formative/causal/refutations/`)**

Called as `result.refute(df)` on a fitted result — always pass the same `df` used for `fit()`. Returns a report object (e.g. `IVRefutationReport`) containing a list of `RefutationCheck` objects, each with `name`, `passed`, and `detail`. Adding a new check means writing a `_check_*` function in the relevant refutations module and appending it to the list in `result.refute()`.

Refutation checks are diagnostics, not proof — the library's philosophy is that no set of tests can guarantee causal validity. Failing checks are signals to investigate, not hard blockers.

**Refutation seeds:** all refutation modules use isolated seeds to avoid collisions with test data (which is typically seeded at 42):
- `_RCC_SEED = 54321` — random common cause column in all refutation modules
- `_PLACEBO_SEED = 99999` — treatment permutation in matching refutation
- `_BOOTSTRAP_SEED = 42` — safe to reuse in matching's bootstrap because it samples row indices from existing data rather than generating independent values that could coincide with columns

Changing `_RCC_SEED` or `_PLACEBO_SEED` to 42 causes the generated noise to collide with instrument/treatment data generated from the same seed, producing a singular matrix in IV.

**Package layout:**
- `formative/causal/dag.py` — `DAG` and `_Node`
- `formative/causal/_exceptions.py` — `IdentificationError`, `GraphError`
- `formative/causal/estimators/ols.py` — `OLSObservational`, `OLSResult` (includes `refute()`)
- `formative/causal/estimators/iv.py` — `IV2SLS`, `IVResult` (includes `refute()`)
- `formative/causal/estimators/matching.py` — `PropensityScoreMatching`, `MatchingResult` (includes `refute()`); also exports `_propensity_scores`, `_att_from_ps` for use by refutations
- `formative/causal/estimators/rct.py` — `RCT`, `RCTResult` (includes `refute()`)
- `formative/causal/estimators/did.py` — `DiD`, `DiDResult` (includes `refute()`)
- `formative/causal/estimators/rdd.py` — `RDD`, `RDDResult` (includes `refute()`)
- `formative/causal/refutations/_check.py` — `Assumption`, `RefutationCheck`, `RefutationReport` (base)
- `formative/causal/refutations/ols.py` — `OLSRefutationReport`, `_check_random_common_cause`
- `formative/causal/refutations/iv.py` — `IVRefutationReport`, `_check_first_stage_f`, `_check_random_common_cause`
- `formative/causal/refutations/matching.py` — `MatchingRefutationReport`, `_check_placebo_treatment`, `_check_random_common_cause`
- `formative/causal/refutations/rct.py` — `RCTRefutationReport`, `_check_random_common_cause`
- `formative/causal/refutations/did.py` — `DiDRefutationReport`, `_check_placebo_group`, `_check_random_common_cause`
- `formative/causal/refutations/rdd.py` — `RDDRefutationReport`, `_check_placebo_cutoff`, `_check_random_common_cause`
- `formative/causal/_explain.py` — narrative rendering for all estimators
- `formative/causal/decision.py` — `DecisionReport` (cost-benefit analysis on a causal estimate)
- `formative/causal/__init__.py` — public API for the causal submodule

**Adding a new estimator:** follow `OLSObservational`/`IV2SLS` — `__init__` validates DAG, `fit()` calls `_identify()`, raises `IdentificationError` where appropriate, returns a result object. Export from `formative/causal/__init__.py`. Add a `refute()` method to the result and a corresponding module under `formative/causal/refutations/`.

**Test pattern for bootstrap-heavy estimators:** matching tests use `setup_class` (not `setup_method`) so `.fit()` runs once per class rather than before every test method. With 500 bootstrap iterations this matters. Matching tests use N=1_000 — larger N causes bootstrap SE to widen while matching's inherent NN noise doesn't shrink at the same rate, which causes refutation checks to fail the 1 SE threshold even on well-specified data.

**Planned estimators:**

The long-term goal is for `formative.causal` to cover all methods in the causal inference decision tree at https://www.maxpagels.com/prototypes/causal-wizard. Remaining methods to add:

- **Synthetic Control** — construct a weighted synthetic control unit from donor units

---

## formative.game

Decision rules for choosing between options under uncertainty. Each rule is a standalone function that accepts `{choice: {scenario: payoff}}` and returns a solver object with a `.solve()` method.

**API pattern:**
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
- `formative/game/__init__.py` — public API for the game submodule

**Adding a new decision rule:** every rule must have docs, tests, and follow these conventions exactly:

- **Code:** a public function `rule_name(outcomes, ...)` that returns a solver class instance. The solver class (`RuleName`) validates inputs in `__init__` and has a `.solve()` method. `.solve()` returns a `RuleNameResult` dataclass. All three are exported from `formative/game/__init__.py`.
- **Result dataclass:** fields are `choice: str`, then any rule-specific scalar for the chosen option (e.g. `guaranteed`, `score`, `expected`), then any dict of all values. Always include a custom `__repr__` that lists all choices, formats the key metric with `{v:+.4g}`, and marks the chosen with `← chosen`.
- **Solver class docstring:** must include an `Examples` block showing `.solve()` and the correct `result.choice` value.
- **Tests:** add a `class TestRuleName` to `tests/test_game.py` (not a separate file). Cover: correct choice, scalar field on result, dict field on result, empty outcomes raises, `← chosen` in repr, and any rule-specific validation.
- **Docs:** add a section to `docs/game/game.rst` with a one-line question header, prose explanation, worked example with manual calculation, a code block, the expected output block, and `.. autoclass::` directives for both the solver and result classes.
