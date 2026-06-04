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

"""Path-pattern rules for per-file VTK conversion.

Provides two small rule systems used by
:class:`~physicsnemo_curator.domains.mesh.sources.vtk.VTKSource`:

* :class:`KeyFilterRule` — per-path-glob data-array include/exclude rules,
  applied at the VTK reader level so unwanted fields are never materialised.
* :func:`resolve_path_value` — resolve a setting (e.g. ``manifold_dim`` or
  ``point_source``) that may be a single scalar applied to every file, or a
  list of ``{"pattern": glob, "value": ...}`` rules selected by file path
  (longest matching pattern wins).

Matching uses :func:`fnmatch.fnmatch` against the full file path string.  When
several patterns match, the longest pattern string is treated as most
specific.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyFilterRule:
    """A single data-array filter rule matched by file-path glob.

    Attributes
    ----------
    path_pattern : str
        Glob pattern matched against the source file path (e.g.
        ``"**/*.vtp"``, ``"**/volume_*.vtu"``).
    mode : {"include", "exclude"}
        ``"include"`` keeps only the listed arrays; ``"exclude"`` drops them.
    keys : list[str]
        Data-array names to include or exclude.
    """

    path_pattern: str
    mode: Literal["include", "exclude"]
    keys: list[str] = field(default_factory=list)


def rules_from_config(raw: list[dict] | None) -> list[KeyFilterRule]:
    """Build :class:`KeyFilterRule` objects from raw config dicts.

    Parameters
    ----------
    raw : list[dict] or None
        Each dict must have ``path_pattern``, ``mode``, and ``keys`` fields.

    Returns
    -------
    list[KeyFilterRule]
        Parsed rules (empty list when *raw* is falsy).
    """
    if not raw:
        return []
    return [
        KeyFilterRule(
            path_pattern=entry["path_pattern"],
            mode=entry["mode"],
            keys=list(entry.get("keys", [])),
        )
        for entry in raw
    ]


def match_filter(file_path: str, rules: list[KeyFilterRule]) -> KeyFilterRule | None:
    """Return the most specific filter rule matching *file_path*.

    Among all rules whose ``path_pattern`` matches, the one with the longest
    pattern string is selected.

    Parameters
    ----------
    file_path : str
        File path to match.
    rules : list[KeyFilterRule]
        Candidate rules.

    Returns
    -------
    KeyFilterRule or None
        The matching rule, or ``None`` if none match.
    """
    matches = [rule for rule in rules if fnmatch.fnmatch(file_path, rule.path_pattern)]
    if not matches:
        return None
    matches.sort(key=lambda r: len(r.path_pattern), reverse=True)
    return matches[0]


def resolve_arrays(
    file_path: str,
    rules: list[KeyFilterRule],
) -> tuple[list[str] | None, list[str] | None]:
    """Resolve ``(include_arrays, exclude_arrays)`` for *file_path*.

    Parameters
    ----------
    file_path : str
        File path to match.
    rules : list[KeyFilterRule]
        Candidate filter rules.

    Returns
    -------
    tuple[list[str] | None, list[str] | None]
        ``(include_arrays, exclude_arrays)`` — at most one is non-``None``.
    """
    rule = match_filter(file_path, rules)
    if rule is None:
        return None, None
    if rule.mode == "include":
        return list(rule.keys), None
    return None, list(rule.keys)


def resolve_path_value(spec: Any, file_path: str, *, default: Any = None) -> Any:
    """Resolve a possibly per-path setting for *file_path*.

    Parameters
    ----------
    spec : Any
        Either a scalar value applied to every file, or a list of
        ``{"pattern": glob, "value": ...}`` rule dicts.
    file_path : str
        File path to match against rule patterns.
    default : Any
        Value returned when *spec* is a rule list but no pattern matches.

    Returns
    -------
    Any
        The resolved value.
    """
    if not isinstance(spec, (list, tuple)):
        return spec

    matched: tuple[int, Any] | None = None
    for entry in spec:
        pattern = entry["pattern"]
        if fnmatch.fnmatch(file_path, pattern) and (matched is None or len(pattern) > matched[0]):
            matched = (len(pattern), entry["value"])
    return matched[1] if matched is not None else default
