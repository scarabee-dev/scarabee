# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Check if we are on Read the Docs server
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# -- Project information -----------------------------------------------------

project = 'Scarabée'
copyright = '2024-2025, Hunter Belanger'
author = 'Hunter Belanger'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.intersphinx',
  'sphinx.ext.autosummary',
  'sphinx.ext.napoleon',
  'sphinx.ext.autosectionlabel',
  'sphinx.ext.todo',
  'sphinx.ext.coverage',
  'sphinx.ext.mathjax',
  'sphinx.ext.ifconfig',
  'sphinx.ext.viewcode',
  'nbsphinx'
]

# Helps reduce duplicate label warnings from autosectionlabel
autosectionlabel_prefix_document = True

autosectionlabel_maxdepth = 4

#suppress_warnings = [
#    'autosectionlabel.*',
#]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'inherited-members': True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

numfig = True

# Mathjax Configuration
mathjax3_config = {
  'loader': {'load': ['[tex]/newcommand', '[tex]/physics', '[tex]/boldsymbol']},
  'tex': {
    'packages': {'[+]': ['newcommand', 'physics', 'boldsymbol']},
    'macros': {
      'pos': '\\boldsymbol{r}',
      'dir': '\\boldsymbol{\hat{\Omega}}',
      'Et': '\\Sigma_t',
      'Es': '\\Sigma_s',
      'Ea': '\\Sigma_a',
      'Ef': '\\Sigma_f',
      'keff': 'k_\\text{eff}'
    }
  }
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#import sphinx_rtd_theme
#html_theme = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']


# Seetings for the PyData theme
html_theme = 'pydata_sphinx_theme'

#html_title = 'Scarabée'
#html_logo = "_images/scarabee_logo.png"
html_favicon = "_images/scarabee_logo.png"

html_theme_options = {
   "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/scarabee-dev/scarabee",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fab fa-github-square",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
   ],

   "logo": {
      "text": "Scarabée",
      "image_light": "_images/scarabee_logo.png",
      "image_dark": "_images/scarabee_logo_dark.png",
   }
}
