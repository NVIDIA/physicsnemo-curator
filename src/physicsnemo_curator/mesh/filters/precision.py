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

"""Precision conversion filter for mesh pipelines.

Converts floating-point field precision (e.g., float64 to float32) to reduce
memory footprint and storage size.  The mesh is modified **in place** and
then yielded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import torch

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)

#: Mapping of string dtype names to torch dtypes.
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _convert_tensordict_precision(
    td: object,
    target_dtype: torch.dtype,
    source_dtypes: tuple[torch.dtype, ...],
    prefix: str = "",
) -> list[str]:
    """Convert floating-point tensor precision in a TensorDict-like object.

    Parameters
    ----------
    td : object
        A TensorDict (``point_data`` or ``cell_data``).
    target_dtype : torch.dtype
        The target dtype to convert to.
    source_dtypes : tuple[torch.dtype, ...]
        Source dtypes to convert from (e.g., (torch.float64,)).
    prefix : str
        Dotted prefix for logging converted fields.

    Returns
    -------
    list[str]
        Fully-qualified key paths that were converted.
    """
    from tensordict import TensorDictBase

    if not isinstance(td, TensorDictBase):
        return []

    converted: list[str] = []

    for key in list(td.keys()):
        child = td[key]
        full_key = f"{prefix}{key}" if prefix else key

        if isinstance(child, TensorDictBase):
            # Recurse into nested TensorDicts
            converted.extend(_convert_tensordict_precision(child, target_dtype, source_dtypes, prefix=f"{full_key}/"))
        elif isinstance(child, torch.Tensor) and child.dtype in source_dtypes:
            # Convert tensor in place
            td[key] = child.to(target_dtype)
            converted.append(full_key)

    return converted


class PrecisionFilter(Filter["Mesh"]):
    """Convert floating-point field precision.

    For each incoming mesh the filter converts all floating-point tensors
    in ``point_data`` and ``cell_data`` from higher precision (e.g., float64)
    to a lower target precision (e.g., float32).  This reduces memory
    footprint and storage size.

    The mesh is modified **in place** and then yielded.

    Parameters
    ----------
    target_dtype : str
        Target dtype: ``"float32"``, ``"float16"``, or ``"bfloat16"``.
        Default is ``"float32"``.

    Examples
    --------
    Convert float64 fields to float32:

    >>> filt = PrecisionFilter(target_dtype="float32")
    >>> pipeline = source.filter(filt).write(sink)

    Convert to half precision for GPU inference:

    >>> filt = PrecisionFilter(target_dtype="float16")
    >>> pipeline = source.filter(filt).write(sink)
    """

    name: ClassVar[str] = "Precision Converter"
    description: ClassVar[str] = "Convert floating-point field precision (e.g., fp64 to fp32)"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for the precision filter.

        Returns
        -------
        list[Param]
            The ``target_dtype`` parameter.
        """
        return [
            Param(
                name="target_dtype",
                description="Target dtype for floating-point fields",
                type=str,
                default="float32",
                choices=["float32", "float16", "bfloat16"],
            ),
        ]

    def __init__(self, target_dtype: str = "float32") -> None:
        if target_dtype not in _DTYPE_MAP:
            msg = f"Unsupported target_dtype: {target_dtype}. Must be one of {list(_DTYPE_MAP.keys())}"
            raise ValueError(msg)

        self._target_dtype = _DTYPE_MAP[target_dtype]
        self._target_dtype_str = target_dtype

        # Determine which source dtypes to convert
        # Only convert from higher precision dtypes
        if target_dtype == "float32":
            self._source_dtypes = (torch.float64,)
        elif target_dtype in ("float16", "bfloat16"):
            self._source_dtypes = (torch.float64, torch.float32)
        else:
            self._source_dtypes = (torch.float64,)

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Convert field precision for each mesh and yield the result.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes.

        Yields
        ------
        Mesh
            The same mesh with fields converted in place.
        """
        for mesh in items:
            all_converted: list[str] = []

            # Convert points tensor if present and applicable
            if mesh.points is not None and mesh.points.dtype in self._source_dtypes:
                mesh.points = mesh.points.to(self._target_dtype)
                all_converted.append("points")

            # Convert point_data fields
            if mesh.point_data is not None:
                all_converted.extend(
                    _convert_tensordict_precision(
                        mesh.point_data,
                        self._target_dtype,
                        self._source_dtypes,
                        prefix="point_data/",
                    )
                )

            # Convert cell_data fields
            if mesh.cell_data is not None:
                all_converted.extend(
                    _convert_tensordict_precision(
                        mesh.cell_data,
                        self._target_dtype,
                        self._source_dtypes,
                        prefix="cell_data/",
                    )
                )

            if all_converted:
                logger.debug(
                    "PrecisionFilter: converted %d field(s) to %s: %s",
                    len(all_converted),
                    self._target_dtype_str,
                    all_converted,
                )

            yield mesh
