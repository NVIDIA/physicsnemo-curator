#!/usr/bin/env python3
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

"""Check that source files contain SPDX license headers.

Scans Python and Rust source files for the required SPDX-FileCopyrightText
and SPDX-License-Identifier headers.  Returns exit code 1 if any file is
missing the required headers.

Usage
-----
    python tests/ci_tests/header_check.py [--all-files]

By default, only files tracked by git are checked.  Pass ``--all-files``
to check every matching file on disk regardless of git status.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

# Required header fragments (must appear within the first 20 lines).
_REQUIRED_FRAGMENTS: list[str] = [
    "SPDX-FileCopyrightText:",
    "SPDX-License-Identifier: Apache-2.0",
]

# File extensions to check.
_EXTENSIONS: set[str] = {".py", ".rs"}

# Directories to skip entirely.
_SKIP_DIRS: set[str] = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "_build",
    "examples-old",
    "benchmarks",
    ".asv",
    "val",
}

# Files to skip (relative to repo root).
_SKIP_FILES: set[str] = {
    "setup.py",
}

# Maximum number of lines to scan in each file.
_SCAN_LINES = 20


def _git_tracked_files(repo_root: pathlib.Path) -> list[pathlib.Path]:
    """Return list of git-tracked files."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=True,
    )
    return [repo_root / line for line in result.stdout.splitlines() if line]


def _walk_all_files(repo_root: pathlib.Path) -> list[pathlib.Path]:
    """Return all matching files on disk."""
    files: list[pathlib.Path] = []
    for ext in _EXTENSIONS:
        files.extend(repo_root.rglob(f"*{ext}"))
    return files


def _should_skip(path: pathlib.Path, repo_root: pathlib.Path) -> bool:
    """Decide whether to skip a file."""
    rel = path.relative_to(repo_root)

    # Skip by directory.
    for part in rel.parts:
        if part in _SKIP_DIRS:
            return True

    # Skip by name.
    if rel.name in _SKIP_FILES:
        return True

    # Skip empty __init__.py files (common pattern).
    return rel.name == "__init__.py" and path.stat().st_size == 0


def _check_file(path: pathlib.Path) -> list[str]:
    """Check a single file for required headers.

    Returns
    -------
    list[str]
        List of missing header fragments, empty if all present.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = "".join(fh.readline() for _ in range(_SCAN_LINES))
    except OSError:
        return [f"Could not read file: {path}"]

    return [frag for frag in _REQUIRED_FRAGMENTS if frag not in head]


def main() -> int:
    """Run the header check and return exit code."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    all_files = "--all-files" in sys.argv

    candidates = _walk_all_files(repo_root) if all_files else _git_tracked_files(repo_root)

    # Filter to relevant extensions.
    candidates = [p for p in candidates if p.suffix in _EXTENSIONS]

    failures: list[tuple[pathlib.Path, list[str]]] = []
    checked = 0

    for path in sorted(candidates):
        if not path.is_file():
            continue
        if _should_skip(path, repo_root):
            continue

        checked += 1
        missing = _check_file(path)
        if missing:
            failures.append((path, missing))

    # Report.
    if failures:
        print(f"\n{'=' * 72}")
        print(f"SPDX header check FAILED — {len(failures)} file(s) missing headers")
        print(f"{'=' * 72}\n")
        for path, missing in failures:
            rel = path.relative_to(repo_root)
            print(f"  {rel}")
            for frag in missing:
                print(f"    - missing: {frag}")
        print(f"\nChecked {checked} files, {len(failures)} failed.\n")
        return 1

    print(f"SPDX header check passed — {checked} files OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
