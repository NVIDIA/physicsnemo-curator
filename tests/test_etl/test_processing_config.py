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


import pytest

from physicsnemo_curator.etl.processing_config import ProcessingConfig


def test_valid_minimal_config():
    """Test creation of ProcessingConfig with minimal valid parameters."""
    config = ProcessingConfig(num_processes=4)
    assert config.num_processes == 4


def test_invalid_num_processes_raises_value_error():
    """Test that invalid num_processes raises ValueError."""
    with pytest.raises(ValueError, match="num_processes must be positive"):
        ProcessingConfig(num_processes=0)

    with pytest.raises(ValueError, match="num_processes must be positive"):
        ProcessingConfig(num_processes=-1)
