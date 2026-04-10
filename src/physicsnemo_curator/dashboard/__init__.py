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

"""Pipeline metrics dashboard.

Launch from the command line::

    curator dashboard pipeline.db

Or from Python::

    from physicsnemo_curator.dashboard import launch
    launch("pipeline.db")

Requires the ``dashboard`` optional dependency group::

    pip install physicsnemo-curator[dashboard]
"""

from __future__ import annotations


def launch(db_path: str, port: int = 5006, open_browser: bool = True) -> None:
    """Launch the pipeline metrics dashboard.

    Parameters
    ----------
    db_path : str
        Path to a PipelineStore SQLite database file.
    port : int
        Port for the Panel server (default 5006).
    open_browser : bool
        Whether to open a browser window automatically.

    Raises
    ------
    ImportError
        If the ``dashboard`` extra is not installed.
    """
    from physicsnemo_curator.dashboard.app import DashboardApp

    app = DashboardApp(db_path)
    app.serve(port=port, open_browser=open_browser)


def __getattr__(name: str) -> object:
    """Lazy import for DashboardApp to avoid requiring panel at import time.

    Parameters
    ----------
    name : str
        Attribute name to look up.

    Returns
    -------
    object
        The requested attribute.

    Raises
    ------
    AttributeError
        If *name* is not a public attribute of this module.
    """
    if name == "DashboardApp":
        from physicsnemo_curator.dashboard.app import DashboardApp

        return DashboardApp
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["DashboardApp", "launch"]
