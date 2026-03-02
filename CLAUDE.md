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

## Architecture

`formative` is a causal inference library built around three layers: DAG, estimators, and refutations.

**Layer 1 — DAG (`formative/dag.py`)**

User builds a `DAG` by calling `dag.assume(node).causes(*effects)`. Edges are validated as acyclic on insertion (Kahn's algorithm). The DAG is the mandatory first input to every estimator.

**Layer 2 — Estimators (`formative/estimators/`)**

Each estimator takes a `DAG` + `treatment`/`outcome` (and optionally an `instrument`) in `__init__`, validates the DAG structure, then at `.fit(df)` time:
- Runs `_identify()` using the backdoor criterion (confounders = common ancestors of treatment and outcome, minus descendants of treatment) to find which variables to control for.
- Raises `IdentificationError` if a DAG-declared confounder is absent from the data and cannot be handled (OLS); IV2SLS does not raise for unobserved confounders since the instrument handles them.
- Fits the model and returns a result object holding both the main estimate and a plain unadjusted OLS estimate for comparison.

**Layer 3 — Refutations (`formative/refutations/`)**

Called as `result.refute(df)` on a fitted result — always pass the same `df` used for `fit()`. Returns a report object (e.g. `IVRefutationReport`) containing a list of `RefutationCheck` objects, each with `name`, `passed`, and `detail`. Adding a new check means writing a `_check_*` function in the relevant refutations module and appending it to the list in `result.refute()`.

Refutation checks are diagnostics, not proof — the library's philosophy (reflected in the docs) is that no set of tests can guarantee causal validity. Failing checks are signals to investigate, not hard blockers.

**Refutation seeds:** all refutation modules use isolated seeds to avoid collisions with test data (which is typically seeded at 42):
- `_RCC_SEED = 54321` — random common cause column in all refutation modules
- `_PLACEBO_SEED = 99999` — treatment permutation in matching refutation
- `_BOOTSTRAP_SEED = 42` — safe to reuse in matching's bootstrap because it samples row indices from existing data rather than generating independent values that could coincide with columns

Changing `_RCC_SEED` or `_PLACEBO_SEED` to 42 causes the generated noise to collide with instrument/treatment data generated from the same seed, producing a singular matrix in IV.

**Package layout:**
- `formative/dag.py` — `DAG` and `_Node`
- `formative/_exceptions.py` — `IdentificationError`, `GraphError`
- `formative/estimators/ols.py` — `OLSObservational`, `OLSResult` (includes `refute()`)
- `formative/estimators/iv.py` — `IV2SLS`, `IVResult` (includes `refute()`)
- `formative/estimators/matching.py` — `PropensityScoreMatching`, `MatchingResult` (includes `refute()`); also exports `_propensity_scores`, `_att_from_ps` for use by refutations
- `formative/estimators/rct.py` — `RCT`, `RCTResult` (includes `refute()`)
- `formative/estimators/did.py` — `DiD`, `DiDResult` (includes `refute()`)
- `formative/refutations/_check.py` — `Assumption`, `RefutationCheck`, `RefutationReport` (base)
- `formative/refutations/ols.py` — `OLSRefutationReport`, `_check_random_common_cause`
- `formative/refutations/iv.py` — `IVRefutationReport`, `_check_first_stage_f`, `_check_random_common_cause`
- `formative/refutations/matching.py` — `MatchingRefutationReport`, `_check_placebo_treatment`, `_check_random_common_cause`
- `formative/refutations/rct.py` — `RCTRefutationReport`, `_check_random_common_cause`
- `formative/refutations/did.py` — `DiDRefutationReport`, `_check_placebo_group`, `_check_random_common_cause`
- `formative/_explain.py` — narrative rendering for all estimators (`explain_ols`, `explain_iv`, `explain_matching`, `explain_rct`, `explain_did`)
- `formative/__init__.py` — public API
- `tests/test_dag.py`, `tests/test_ols.py`, `tests/test_iv.py`, `tests/test_matching.py`, `tests/test_rct.py`, `tests/test_did.py`, `tests/test_ols_refutation.py`, `tests/test_iv_refutation.py`, `tests/test_matching_refutation.py`
- `examples/ols/`, `examples/iv/`, `examples/matching/`, `examples/rct/`, `examples/did/` — runnable examples

**Adding a new estimator:** follow `OLSObservational`/`IV2SLS` — `__init__` validates DAG, `fit()` calls `_identify()`, raises `IdentificationError` where appropriate, returns a result object. Export from `formative/__init__.py`. Add a `refute()` method to the result and a corresponding module under `formative/refutations/`.

**Test pattern for bootstrap-heavy estimators:** matching tests use `setup_class` (not `setup_method`) so `.fit()` runs once per class rather than before every test method. With 500 bootstrap iterations this matters. Matching tests use N=1_000 — larger N causes bootstrap SE to widen while matching's inherent NN noise doesn't shrink at the same rate, which causes refutation checks to fail the 1 SE threshold even on well-specified data.

## Planned estimators

The long-term goal is for `formative` to cover all methods in the causal inference decision tree at https://www.maxpagels.com/prototypes/causal-wizard. Remaining methods to add:

- **RCT** (Randomized Controlled Trial) — ✓ implemented as `RCT` in `formative/estimators/rct.py`
- **DiD** (Difference-in-Differences) — ✓ implemented as `DiD` in `formative/estimators/did.py`
- **RD** (Regression Discontinuity) — treatment assigned by a threshold on a running variable
- **IV** (Instrumental Variables) — ✓ implemented as `IV2SLS` in `formative/estimators/iv.py`
- **Matching** — ✓ implemented as `PropensityScoreMatching` in `formative/estimators/matching.py`
- **Synthetic Control** — construct a weighted synthetic control unit from donor units
