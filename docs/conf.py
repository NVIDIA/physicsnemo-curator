# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sphinx configuration for physicsnemo-curator documentation."""

import os

project = "physicsnemo-curator"
copyright = "2025 - 2026, NVIDIA CORPORATION & AFFILIATES"  # noqa: A001
author = "NVIDIA"
version = "0.1.0"
release = "0.1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "autoapi.extension",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# ---------------------------------------------------------------------------
# Sphinx-Gallery (executable examples)
# ---------------------------------------------------------------------------
# Set PLOT_GALLERY=0 to skip example execution during doc builds.
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"/.+\.py$",
    "run_stale_examples": os.environ.get("RUN_STALE_EXAMPLES", "False").lower() in ("1", "true"),
    "plot_gallery": os.environ.get("PLOT_GALLERY", "0"),
}

# ---------------------------------------------------------------------------
# Autosummary
# ---------------------------------------------------------------------------
autosummary_generate = False  # we use autosummary for compact tables only

# ---------------------------------------------------------------------------
# AutoAPI (static analysis — reads .pyi stubs without importing)
# ---------------------------------------------------------------------------
autoapi_dirs = ["../src/physicsnemo_curator"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# ---------------------------------------------------------------------------
# Napoleon (NumPy-style docstrings)
# ---------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# ---------------------------------------------------------------------------
# MyST (Markdown support)
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

# ---------------------------------------------------------------------------
# Intersphinx (cross-references to external docs)
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pyvista": ("https://docs.pyvista.org/version/stable/", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]

# PyData Sphinx Theme options (nvidia-sphinx-theme extends pydata-sphinx-theme)
html_theme_options = {
    "github_url": "https://github.com/NVIDIA/physicsnemo-curator",
    "navbar_align": "content",
    "navbar_start": ["navbar-logo", "navbar-nav"],
    "navbar_center": [],
}

# ---------------------------------------------------------------------------
# Source settings
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
