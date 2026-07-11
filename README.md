# formative

Python library for quantitative reasoning.

## Requirements

- Python 3.10+

## Installation

```bash
pip install formative-ds
```

## Docs

Comprehensive documentation is available at [docs.getformative.dev](https://docs.getformative.dev).

## Usage

### Causal estimation

Every analysis follows the same four steps: assume, estimate, refute, decide.

```python
from formative.causal import DAG, OLSObservational

# 1. Encode your causal assumptions as a DAG
dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

# 2. Estimate the causal effect
result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
print(result.summary())

# 3. Refute: stress-test the result's assumptions
print(result.refute(df).summary())

# 4. Decide: is the treatment worth acting on?
print(result.decide(cost=8, benefit=15))
```

Confounders declared in the DAG are controlled for automatically. If a confounder is absent from the dataframe, an `IdentificationError` is raised before any estimation runs. Some estimators go further at step 4 — per-group decisions, or learning a treatment rule with `learn_policy()`.

### Decision rules

```python
from formative.game import maximin, maximax, hurwicz, laplace, minimax

outcomes = {
    "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
    "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
    "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
}

maximin(outcomes).solve()        # safest choice (best worst case)
maximax(outcomes).solve()        # most optimistic (best best case)
hurwicz(outcomes, alpha=0.5).solve()  # blend of optimism and pessimism
laplace(outcomes).solve()        # highest average payoff
minimax(outcomes).solve()        # lowest worst-case regret
```

See online documentation at [docs.getformative.dev](https://docs.getformative.dev) for more examples and details.

## Local development

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/maxpagels/formative
cd formative
uv sync --dev
```

This creates a `.venv`, installs all dependencies, and installs the package in editable mode.

### Releasing a new version

```bash
make release BUMP=patch   # 0.1.0 → 0.1.1 (bug fixes)
make release BUMP=minor   # 0.1.0 → 0.2.0 (new features)
make release BUMP=major   # 0.1.0 → 1.0.0 (breaking changes)
```

One command does everything: bumps the version in `pyproject.toml` and `uv.lock`
(commit + tag), builds the docs, snapshots them into `site/<major.minor>/` (the versioned
docs site Vercel serves statically), and pushes with tags — which triggers the publish to
PyPI. It refuses to run if the working tree is dirty or `uv.lock` is out of date.

### Running tests

```bash
uv run pytest
```

### Importing without installing

To use `formative` from a script outside this repo without installing it, either prepend the path at runtime:

```python
import sys
sys.path.insert(0, "/path/to/formative")

from formative.causal import DAG, OLSObservational
```

Or set `PYTHONPATH` before running:

```bash
PYTHONPATH=/path/to/formative python your_script.py
```
