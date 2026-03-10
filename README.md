# formative

Python library for causal effect estimation and game theory.

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

```python
from formative import DAG, OLSObservational

dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

result = OLSObservational(
    dag,
    treatment="education",
    outcome="income"
).fit(df)

print(result.summary())
```

Confounders declared in the DAG are controlled for automatically. If a confounder is absent from the dataframe, an `IdentificationError` is raised before any estimation runs.

### Decision rules

```python
from formative.game import maximin, maximax, hurwicz, laplace, minimax_regret

outcomes = {
    "stocks": {"recession": -20, "stagnation":  5, "growth": 30},
    "bonds":  {"recession":   5, "stagnation":  5, "growth":  7},
    "cash":   {"recession":   2, "stagnation":  2, "growth":  2},
}

maximin(outcomes).solve()        # safest choice (best worst case)
maximax(outcomes).solve()        # most optimistic (best best case)
hurwicz(outcomes, alpha=0.5).solve()  # blend of optimism and pessimism
laplace(outcomes).solve()        # highest average payoff
minimax_regret(outcomes).solve() # lowest worst-case regret
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
uvx bump-my-version bump patch   # 0.1.0 → 0.1.1 (bug fixes)
uvx bump-my-version bump minor   # 0.1.0 → 0.2.0 (new features)
uvx bump-my-version bump major   # 0.1.0 → 1.0.0 (breaking changes)
git push --follow-tags            # triggers publish to PyPI
```

### Running tests

```bash
uv run pytest
```

### Importing without installing

To use `formative` from a script outside this repo without installing it, either prepend the path at runtime:

```python
import sys
sys.path.insert(0, "/path/to/formative")

from formative import DAG, OLSObservational
```

Or set `PYTHONPATH` before running:

```bash
PYTHONPATH=/path/to/formative python your_script.py
```
