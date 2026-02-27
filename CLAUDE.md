# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies and the package in editable mode
uv sync --dev

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_ols.py
uv run pytest tests/test_ols.py::TestDAG::test_basic_edges

# Build docs
make build-docs
# or directly:
uv run --group docs sphinx-build -b html docs docs/_build
```

## Architecture

`formative` is a causal inference library that enforces explicit causal assumptions via a DAG before any estimation runs.

**Core flow:**
1. User builds a `DAG` by calling `dag.assume(node).causes(*effects)` — edges are validated as acyclic on insertion (Kahn's algorithm).
2. An estimator (`OLSObservational`) takes the DAG plus `treatment`/`outcome` variable names. At construction it validates that both variables are nodes in the DAG.
3. At `.fit(df)` time, the estimator runs `_identify()` which applies the backdoor criterion: confounders = ancestors of both treatment and outcome, minus descendants of treatment. Any confounder in the DAG that's absent from the dataframe raises `IdentificationError` before any regression runs.
4. OLS is fit twice — once adjusted (controlling for the identified confounders) and once unadjusted — and both are returned in `OLSResult`.

**Package layout:**
- `formative/dag.py` — `DAG` and `_Node` (proxy returned by `assume()`)
- `formative/_exceptions.py` — `IdentificationError`, `GraphError`
- `formative/estimators/ols.py` — `OLSObservational` (estimator) and `OLSResult`
- `formative/__init__.py` — public API: `DAG`, `OLSObservational`, `OLSResult`
- `tests/test_ols.py` — all tests (covers DAG properties and OLS estimation)

**Adding a new estimator** follows the same pattern as `OLSObservational`: accept a `DAG` + treatment/outcome in `__init__`, call `_identify()` (or equivalent) in `fit()`, raise `IdentificationError` for unobserved DAG confounders, return a result object. Export it from `formative/__init__.py`.

## Planned estimators

The long-term goal is for `formative` to cover all methods in the causal inference decision tree at https://www.maxpagels.com/prototypes/causal-wizard. Currently only observational OLS is implemented. Remaining methods to add:

- **RCT** (Randomized Controlled Trial) — treatment is randomised, no confounding adjustment needed
- **DiD** (Difference-in-Differences) — panel/repeated-measures data with a treatment group and control group
- **RD** (Regression Discontinuity) — treatment assigned by a threshold on a running variable
- **IV** (Instrumental Variables) — ✓ implemented as `IV2SLS` in `formative/estimators/iv.py`
- **Matching** — match treated and control units on observed confounders
- **Synthetic Control** — construct a weighted synthetic control unit from donor units
