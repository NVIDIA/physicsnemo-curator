# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Pytest configuration for physicsnemo_curator tests."""

import multiprocessing


def pytest_configure(config):
    """Configure pytest and set multiprocessing start method.

    This fixes the Python 3.12+ warning:
    "This process is multi-threaded, use of fork() may lead to deadlocks in the child."

    The 'spawn' method is safer for multi-threaded applications (numpy, zarr, etc.)
    than 'fork' on Linux/Unix systems.
    """
    try:
        # Set the start method to 'spawn' instead of 'fork'
        # This must be called before any multiprocessing code runs
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method was already set, which is fine
        pass
