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

"""Import-time benchmarks for the physicsnemo-curator package."""


def timeraw_import_curator():
    """Measure import time of the physicsnemo_curator package in a fresh subprocess."""
    return "import physicsnemo_curator"


def timeraw_import_curator_core():
    """Measure import time of physicsnemo_curator.core in a fresh subprocess."""
    return "from physicsnemo_curator.core import base"
