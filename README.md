# formative

Python package for causal effect estimation. Forces you to encode your causal assumptions as a DAG before choosing an estimation method — making identification explicit rather than implicit.

## Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
git clone https://github.com/maxpagels/formative
cd formative
uv sync --dev
```

This creates a `.venv`, installs all dependencies, and installs the package itself in editable mode.

## Running tests

```bash
uv run pytest
```

## Usage

```python
from formative import DAG, OLSObservational

# 1. Encode your causal assumptions
dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

# 2. Estimate — confounders in the DAG are controlled for automatically
#    if they appear in df. If they don't, an IdentificationError is raised.
result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
print(result.summary())
```

If a confounder in the DAG is absent from the dataframe, the package raises an `IdentificationError` before any estimation runs:

```python
# df does not contain an "ability" column — ability treated as unobserved

result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
# IdentificationError: Unobserved confounders detected: ['ability']
# These variables influence both 'education' and 'income'
# but are not in the dataframe and cannot be controlled for.
```

## Importing locally without installing

If you want to import `formative` from a script outside this repo without installing it:

```python
import sys
sys.path.insert(0, "/path/to/formative")

from formative import DAG, OLSObservational
```

Or set `PYTHONPATH` before running your script:

```bash
PYTHONPATH=/path/to/formative python your_script.py
```

The cleanest option is to use `uv sync --dev` inside the repo, which installs the package in editable mode into `.venv`. Any script run via `uv run` within the repo will pick it up automatically.
