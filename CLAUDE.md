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

**Refutation seed:** both `formative/refutations/ols.py` and `formative/refutations/iv.py` use `_RCC_SEED = 54321` for the random common cause column. This must not be changed to a common test seed (e.g. 42) — doing so causes the generated noise to collide with instrument/treatment data generated from the same seed, producing a singular matrix in IV.

**Package layout:**
- `formative/dag.py` — `DAG` and `_Node`
- `formative/_exceptions.py` — `IdentificationError`, `GraphError`
- `formative/estimators/ols.py` — `OLSObservational`, `OLSResult` (includes `refute()`)
- `formative/estimators/iv.py` — `IV2SLS`, `IVResult` (includes `refute()`)
- `formative/refutations/_check.py` — `RefutationCheck`
- `formative/refutations/ols.py` — `OLSRefutationReport`, `_check_random_common_cause`
- `formative/refutations/iv.py` — `IVRefutationReport`, `_check_first_stage_f`, `_check_random_common_cause`
- `formative/__init__.py` — public API
- `tests/test_dag.py`, `tests/test_ols.py`, `tests/test_iv.py`, `tests/test_ols_refutation.py`, `tests/test_iv_refutation.py`
- `examples/ols/`, `examples/iv/` — runnable examples

**Adding a new estimator:** follow `OLSObservational`/`IV2SLS` — `__init__` validates DAG, `fit()` calls `_identify()`, raises `IdentificationError` where appropriate, returns a result object. Export from `formative/__init__.py`. Add a `refute()` method to the result and a corresponding module under `formative/refutations/`.

## Planned estimators

The long-term goal is for `formative` to cover all methods in the causal inference decision tree at https://www.maxpagels.com/prototypes/causal-wizard. Remaining methods to add:

- **RCT** (Randomized Controlled Trial) — treatment is randomised, no confounding adjustment needed
- **DiD** (Difference-in-Differences) — panel/repeated-measures data with a treatment group and control group
- **RD** (Regression Discontinuity) — treatment assigned by a threshold on a running variable
- **IV** (Instrumental Variables) — ✓ implemented as `IV2SLS` in `formative/estimators/iv.py`
- **Matching** — match treated and control units on observed confounders
- **Synthetic Control** — construct a weighted synthetic control unit from donor units
