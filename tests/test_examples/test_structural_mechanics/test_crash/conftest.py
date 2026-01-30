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

"""Test configuration for crash simulation tests.

This conftest.py adds the crash example directory to sys.path when this
package is imported by pytest. This allows test modules to import crash
example modules without PYTHONPATH manipulation.

To prevent cross-contamination, this also removes other example directories
from sys.path before adding the crash directory.
"""

import os
import sys

# Calculate paths
_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_test_dir, "../../../.."))
_examples_dir = os.path.join(_repo_root, "examples")
_crash_dir = os.path.join(_repo_root, "examples/structural_mechanics/crash")

# Remove any other example directories from sys.path to prevent cross-contamination
# This is important when running multiple example test suites together
_paths_to_remove = [
    p for p in sys.path if p.startswith(_examples_dir) and p != _crash_dir
]
for path in _paths_to_remove:
    sys.path.remove(path)

# Add crash example directory to the VERY FRONT of sys.path
# This ensures it's checked before any other paths
if _crash_dir in sys.path:
    sys.path.remove(_crash_dir)
sys.path.insert(0, _crash_dir)
