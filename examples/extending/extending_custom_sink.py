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

"""
Creating a Custom Sink
=======================

This example shows how to implement and register a custom
:class:`~physicsnemo_curator.core.base.Sink`.

We create an ``HDF5Sink`` that writes :class:`xarray.DataArray` fields
to HDF5 files — one file per source index, with each variable stored
as a separate dataset.  This demonstrates the core sink contract:
consume an iterator of items, persist them, and return the paths of
written files.

.. note::

   Install the DataArray extras and h5py before running::

       pip install physicsnemo-curator[da] h5py
"""

# %%
# Step 1 — Define the Sink
# -------------------------
#
# A sink inherits from :class:`~physicsnemo_curator.core.base.Sink` and
# implements three things:
#
# 1. ``name`` / ``description`` class variables
# 2. ``params()`` class method
# 3. ``__call__(items, index)`` — consume the iterator, write data,
#    return a list of written file paths

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Param, Sink

if TYPE_CHECKING:
    from collections.abc import Iterator

    import xarray as xr


class HDF5Sink(Sink["xr.DataArray"]):
    """Write DataArrays to HDF5 files with per-variable datasets.

    Creates one ``.h5`` file per pipeline index.  Each incoming
    DataArray is split along the ``variable`` dimension (if present)
    and stored as a separate HDF5 dataset.

    Parameters
    ----------
    output_dir : str
        Directory where ``.h5`` files are written.
    compression : str
        HDF5 compression filter (e.g. ``"gzip"``, ``"lzf"``).
    compression_level : int
        Compression level (0 = off, 9 = max for gzip).
    """

    name: ClassVar[str] = "HDF5 Writer"
    description: ClassVar[str] = "Write DataArrays to HDF5 files"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors for this sink.

        Returns
        -------
        list[Param]
            Parameters: output_dir, compression, compression_level.
        """
        return [
            Param(name="output_dir", description="Output directory for HDF5 files", type=str),
            Param(
                name="compression",
                description="HDF5 compression filter",
                type=str,
                default="gzip",
                choices=["gzip", "lzf"],
            ),
            Param(
                name="compression_level",
                description="Compression level (0=off, 9=max)",
                type=int,
                default=4,
            ),
        ]

    def __init__(
        self,
        output_dir: str,
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._compression = compression
        self._compression_level = compression_level

    def __call__(self, items: Iterator[xr.DataArray], index: int) -> list[str]:
        """Consume DataArrays and write to an HDF5 file.

        Parameters
        ----------
        items : Iterator[xr.DataArray]
            Stream of DataArray items to persist.
        index : int
            Source index (used for naming the output file).

        Returns
        -------
        list[str]
            Paths of written files.
        """
        import h5py

        self._output_dir.mkdir(parents=True, exist_ok=True)
        h5_path = self._output_dir / f"data_{index:04d}.h5"

        paths: list[str] = []
        with h5py.File(str(h5_path), "w") as f:
            written = False
            for da in items:
                if "variable" in da.dims:
                    for var_name in da.coords["variable"].values:
                        var_da = da.sel(variable=var_name).drop_vars("variable")
                        ds = f.create_dataset(
                            str(var_name),
                            data=var_da.values,
                            compression=self._compression,
                            compression_opts=self._compression_level if self._compression == "gzip" else None,
                        )
                        # Store coordinate metadata as attributes
                        for dim in var_da.dims:
                            ds.attrs[f"dim_{dim}"] = str(dim)
                        written = True
                else:
                    f.create_dataset(
                        "data",
                        data=da.values,
                        compression=self._compression,
                        compression_opts=self._compression_level if self._compression == "gzip" else None,
                    )
                    written = True

            if written:
                paths.append(str(h5_path))

        # If nothing was written, remove the empty file
        if not paths and h5_path.exists():
            h5_path.unlink()

        return paths


# %%
# Step 2 — Register the Sink (Optional)
# ----------------------------------------
#
# Registration makes the sink discoverable in the global registry.

from physicsnemo_curator.core.registry import registry

registry.register_sink("da", HDF5Sink)

registered = registry.sinks("da")
print(f"Registered DA sinks: {list(registered.keys())}")
assert "HDF5 Writer" in registered

# %%
# Step 3 — Use in a Pipeline
# ---------------------------
#
# The custom sink plugs into the standard pipeline API.  We fetch
# ERA5 temperature and wind data, then write each timestep to a
# separate HDF5 file.

from datetime import datetime

from physicsnemo_curator.domains.da.sources.era5 import ERA5Source
from physicsnemo_curator.run import run_pipeline

source = ERA5Source(
    times=[datetime(2020, 6, 1, 0), datetime(2020, 6, 1, 6)],
    variables=["t2m", "u10m"],
    backend="arco",
)

pipeline = source.write(HDF5Sink(output_dir="outputs/extending/hdf5/"))

print(f"Source items: {len(pipeline)}")

results = run_pipeline(
    pipeline,
    n_jobs=1,
    backend="sequential",
    indices=range(len(pipeline)),
    progress=True,
)

print(f"\nProcessed {len(results)} items")
for i, paths in enumerate(results):
    print(f"  Index {i}: {paths}")

# %%
# Step 4 — Verify Output
# -----------------------
#
# Read back the HDF5 file to confirm the data was written correctly.

import h5py

first_path = results[0][0]
with h5py.File(first_path, "r") as f:
    print(f"\nHDF5 datasets in {first_path}:")
    for key in f:
        ds = f[key]
        print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")

# %%
# Summary
# -------
#
# To create a custom sink:
#
# 1. Subclass :class:`~physicsnemo_curator.core.base.Sink` with a
#    type parameter (``Sink["xr.DataArray"]``, ``Sink["Mesh"]``, etc.)
# 2. Set ``name`` and ``description`` class variables
# 3. Implement ``params()`` and ``__call__(items, index) -> list[str]``
# 4. Ensure the output directory is created automatically
# 5. Return ``[]`` for empty iterators (no crash, no empty files)
# 6. Optionally register with ``registry.register_sink()``
#
# For **append** semantics (multiple indices writing to the same file),
# see :class:`~physicsnemo_curator.domains.da.sinks.zarr_writer.ZarrSink`.
