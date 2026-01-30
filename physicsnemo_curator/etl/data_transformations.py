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

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from physicsnemo_curator.etl.processing_config import ProcessingConfig


class DataTransformation(ABC):
    """Abstract base class for data transformations."""

    @abstractmethod
    def __init__(self, cfg: ProcessingConfig):
        self.config = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def transform(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Transform input data."""
        pass
