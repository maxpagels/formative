# TODO

## Remove built docs from version control

`docs/_build/` (90 files, ~11MB of generated HTML, pickles, static assets) is tracked in git.
`vercel.json` already has a `buildCommand` that runs `sphinx-build` on deploy, so the output
does not need to be committed.

1. Add `docs/_build/` to `.gitignore`
2. Remove it from the index: `git rm -r --cached docs/_build/`

## Centralise refutation seeds

`_RCC_SEED = 54321` and `_PLACEBO_SEED = 99999` are duplicated across all six refutation
modules (`ols.py`, `iv.py`, `rct.py`, `matching.py`, `did.py`, `rdd.py`). Move them to
`formative/refutations/_check.py` and import from there.

## Publish to PyPI

The README installation section currently has a placeholder `pip install formative` with a
note that it is not yet published.
