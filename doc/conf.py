# -*- coding: utf-8 -*-
#
# Pastas documentation build configuration file, created by
# sphinx-quickstart on Wed May 11 12:38:06 2016.

import os
import re
import sys
from datetime import date

import requests

year = date.today().strftime("%Y")

from matplotlib import use

use("agg")

from pastas import __version__

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("."))

# -- Load extensions ------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",  # lowercase didn't work
    "numpydoc",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
]

# -- General configuration ------------------------------------------------------------

templates_path = ["_templates"]
source_suffix = ".rst"
source_encoding = "utf-8"

master_doc = "index"  # The master toctree document.

# General information about the project.
project = "Pastas"
copyright = "{}, The Pastas Team".format(year)
author = "R.A. Collenteur, M. Bakker, R. Calje, F. Schaars"

# The version.
version = __version__
release = __version__
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "**groundwater_paper",
    "**.ipynb_checkpoints",
]

add_function_parentheses = False
add_module_names = False
show_authors = False  # section and module author directives will not be shown
todo_include_todos = False  # Do not show TODOs in docs

# -- Options for HTML output ----------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_short_title = "Pastas"
html_favicon = "_static/favo.ico"
html_css_files = ["css/custom.css"]
html_show_sphinx = True
html_show_copyright = True
htmlhelp_basename = "Pastasdoc"  # Output file base name for HTML help builder.
html_use_smartypants = True
html_show_sourcelink = True

html_theme_options = {
    "github_url": "https://github.com/pastas/pastas",
    "use_edit_page_button": True,
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",  # Label for this link
            "url": "https://github.com/pastas/pastas",  # required
            "icon": "fab fa-github-square",
            "type": "fontawesome",  # Default is fontawesome
        }
    ],
}

html_context = {
    "github_user": "pastas",
    "github_repo": "pastas",
    "github_version": "master",
    "doc_path": "doc",
}

# -- Napoleon settings ----------------------------------------------------------------

napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_type_aliases = {
    "array-like": ":term:`array-like <array_like>`",
    "array_like": ":term:`array_like`",
    "ps": "pastas",
    "ml": "pastas.model.Model",
    "TimestampType": "pandas.Timestamp",
}

# -- Autodoc, autosummary, and autosectionlabel settings ------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"

autosummary_generate = True

autoclass_content = "class"

autosectionlabel_prefix_document = True

# -- Numpydoc settings ----------------------------------------------------------------

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

# -- Generating references and publications lists with bibtex -------------------------

# Get a Bibtex reference file from the Zotero group for referencing
url = "https://api.zotero.org/groups/4846685/collections/8UG7PVLY/items/"
params = {"format": "bibtex", "style": "apa", "limit": 100}

r = requests.get(url=url, params=params)
with open("about/references.bib", mode="w", encoding="utf-8") as file:
    file.write(r.text)

# Get a Bibtex reference file from the Zotero group for publications list
url = "https://api.zotero.org/groups/4846685/collections/Q4F7R59G/items/"
params = {"format": "bibtex", "style": "apa", "limit": 100}

r = requests.get(url=url, params=params)
with open("about/publications.bib", mode="w", encoding="utf-8") as file:
    # Replace citation key to prevent duplicate labels and article now shown
    text = re.sub(r"(@([a-z]*){)", r"\1X_", r.text)
    file.write(text)

# Add some settings for bibtex
bibtex_bibfiles = ["about/references.bib", "about/publications.bib"]
bibtex_reference_style = "author_year"

# -- Set intersphinx Directories ------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- myst_nb options ------------------------------------------------------------------

nb_execution_allow_errors = True  # Allow errors in notebooks, to see the error online
nb_execution_mode = "auto"
