# formative

Python library for causal effect estimation. You declare your causal assumptions as a DAG before choosing an estimation method, making identification explicit rather than implicit.

## Requirements

- Python 3.11+

## Installation

```bash
pip install formative-ds
```

## Docs

Comprehensive documentation is available at [docs.getformative.dev](https://docs.getformative.dev).

## Usage

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

Confounders declared in the DAG are controlled for automatically. If a confounder is absent from the dataframe, an `IdentificationError` is raised before any estimation runs. See online documentation at [docs.getformative.dev](https://docs.getformative.dev) for more examples and details.

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
