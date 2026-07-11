import os
import re
import sys

# Make the package importable without installing
sys.path.insert(0, os.path.abspath(".."))

project = "formative"
copyright = "2025, Max Pagels"
author = "Max Pagels"

# Read the version from pyproject.toml (the bump-my-version source of truth).
# Not importlib.metadata: the editable install's recorded version goes stale
# right after a bump, before the next `uv sync`.
with open(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")) as _f:
    release = re.search(r'^version = "([^"]+)"', _f.read(), re.M).group(1)
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # NumPy / Google docstring styles
    "sphinx_autodoc_typehints",  # render type hints from annotations
    "sphinx_copybutton",  # copy button on code blocks
]

html_theme = "sphinx_rtd_theme"
# Canonical URLs point at the unversioned paths, which the Vercel rewrite
# serves from the latest snapshot — search engines index one copy per page.
html_baseurl = "https://docs.getformative.dev/"
html_static_path = ["_static"]
html_css_files = ["version_switcher.css"]
html_js_files = ["version_switcher.js"]

# autodoc: show members in source order, include type hints in signatures
autodoc_member_order = "bysource"
autodoc_typehints = "description"
always_document_param_types = True

# napoleon: we use plain reStructuredText docstrings with Parameters sections
napoleon_use_param = True
napoleon_use_rtype = False
