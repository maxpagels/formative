# formative

Python library for causal effect estimation. You declare your causal assumptions as a DAG before choosing an estimation method, making identification explicit rather than implicit.

## Requirements

- Python 3.11+

## Installation

```bash
pip install formative
```

(Not yet published to PyPI.)

### Local development

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/maxpagels/formative
cd formative
uv sync --dev
```

This creates a `.venv`, installs all dependencies, and installs the package in editable mode.

## Usage

```python
from formative import DAG, OLSObservational

dag = DAG()
dag.assume("ability").causes("education", "income")
dag.assume("education").causes("income")

result = OLSObservational(dag, treatment="education", outcome="income").fit(df)
print(result.summary())
```

Confounders declared in the DAG are controlled for automatically. If a confounder is absent from the dataframe, an `IdentificationError` is raised before any estimation runs.

## Refutations

Every result object exposes a `.refute(df)` method that runs a set of diagnostic checks. Pass the same dataframe used for `.fit()`.

```python
report = result.refute(df)
print(report.summary())
```

Each check in the report has a `name`, a `passed` flag, and a `detail` string. Failing checks are signals to investigate, not hard blockers. No set of refutation tests can guarantee causal validity.

## Running tests

```bash
uv run pytest
```

## Importing without installing

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
