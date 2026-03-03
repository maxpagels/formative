# TODO

## Remove built docs from version control

`docs/_build/` (90 files, ~11MB of generated HTML, pickles, static assets) is tracked in git.
`vercel.json` already has a `buildCommand` that runs `sphinx-build` on deploy, so the output
does not need to be committed.

1. Add `docs/_build/` to `.gitignore`
2. Remove it from the index: `git rm -r --cached docs/_build/`

## Centralise refutation seeds and constants

`_RCC_SEED = 54321`, `_PLACEBO_SEED = 99999`, and `_RCC_COL = "_rcc"` are copy-pasted into
every refutation module. Move them to `formative/refutations/_check.py` and import from there.

## Extract the shift-check tail of _check_random_common_cause

Every `_check_random_common_cause` ends with identical code: compute `shift`, compare to
`original_se`, format the detail string, return a `RefutationCheck`. Only the model-fitting
in the middle differs per method. Extract the tail into a helper in `_check.py`:

```python
def _shift_check(shift: float, se: float, failure_note: str) -> RefutationCheck: ...
```

Each module then calls it after computing `new_effect`.

## Unify DAG section rendering in _explain.py

`_dag_section()` exists and is used by `explain_ols`, `explain_iv`, and `explain_matching`,
but `explain_did` and `explain_rdd` inline the same DAG-edge loop manually. Replace the
inline blocks with a call to `_dag_section()`.

## Extract node-membership validation

Every estimator `_validate_inputs()` contains the same loop:

```python
for label, var in [("Treatment", self._treatment), ("Outcome", self._outcome)]:
    if var not in nodes:
        raise ValueError(f"{label} '{var}' is not a node in the DAG. ...")
```

Extract this to a helper (e.g., `dag.assert_nodes_exist({"Treatment": t, "Outcome": y})`)
so each estimator calls one line instead of repeating the loop.

## Publish to PyPI

The README installation section currently has a placeholder `pip install formative` with a
note that it is not yet published.
