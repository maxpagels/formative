import os
import sys

# Make the package importable without installing
sys.path.insert(0, os.path.abspath(".."))

project   = "formative"
copyright = "2025, Max Pagels"
author    = "Max Pagels"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       # NumPy / Google docstring styles
    "sphinx_autodoc_typehints",  # render type hints from annotations
    "sphinx_copybutton",         # copy button on code blocks
]

html_theme = "sphinx_rtd_theme"

# autodoc: show members in source order, include type hints in signatures
autodoc_member_order    = "bysource"
autodoc_typehints       = "description"
always_document_param_types = True

# napoleon: we use plain reStructuredText docstrings with Parameters sections
napoleon_use_param  = True
napoleon_use_rtype  = False
