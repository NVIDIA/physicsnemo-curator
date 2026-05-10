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

"""Creating a Custom Sink.

See README.md for a full walkthrough of this example.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, ClassVar

from physicsnemo_curator.core.base import Param, Sink, Source

if TYPE_CHECKING:
    from collections.abc import Iterator

    import xarray as xr


# Step 1 — Define the Sink


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

        if not paths and h5_path.exists():
            h5_path.unlink()

        return paths


# Step 2 — Register the Sink (Optional)

import physicsnemo_curator.domains.da  # noqa: F401 - registers "da" submodule
from physicsnemo_curator.core.registry import registry

registry.register_sink("da", HDF5Sink)

registered = registry.sinks("da")
print(f"Registered DA sinks: {list(registered.keys())}")
assert "HDF5 Writer" in registered

# Step 3 — Use in a Pipeline

from physicsnemo_curator.domains.da.sources.random import RandomDataArraySource
from physicsnemo_curator.run import run_pipeline

source = RandomDataArraySource(
    n_samples=4,
    n_lat=32,
    n_lon=64,
    variables="t2m,u10m",
    seed=123,
)

pipeline = source.write(HDF5Sink(output_dir="output/extending/hdf5/"))

print(f"Source items: {len(pipeline)}")

results = run_pipeline(
    pipeline,
    n_jobs=1,
    backend="sequential",
    indices=range(len(pipeline)),
    use_tui=True,
)

print(f"\nProcessed {len(results)} items")
for i, paths in enumerate(results):
    print(f"  Index {i}: {paths}")

# Step 4 — Verify Output

import h5py

first_path = results[0][0]
with h5py.File(first_path, "r") as f:
    print(f"\nHDF5 datasets in {first_path}:")
    for key in f:
        ds = f[key]
        print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")


# Extended API: set_source()


class TemplatedHDF5Sink(Sink["xr.DataArray"]):
    """A sink that uses source metadata for output naming.

    Demonstrates the set_source() pattern for resolving naming
    placeholders from the source's file structure.

    Parameters
    ----------
    output_dir : str
        Base output directory.
    naming_template : str
        Template for subdirectory names.  Supports placeholders:
        ``{index}``, ``{relpath}``, ``{stem}``.
    """

    name: ClassVar[str] = "Templated HDF5 Writer"
    description: ClassVar[str] = "Write HDF5 with source-aware naming"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors.

        Returns
        -------
        list[Param]
            Parameters: output_dir and naming_template.
        """
        return [
            Param(name="output_dir", description="Base output directory", type=str),
            Param(
                name="naming_template",
                description="Naming template with {index}, {relpath}, {stem} placeholders",
                type=str,
                default="{index:04d}",
            ),
        ]

    def __init__(self, output_dir: str, naming_template: str = "{index:04d}") -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._naming_template = naming_template
        self._source: Source | None = None

    def set_source(self, source: Source) -> None:
        """Receive the pipeline source for placeholder resolution.

        Parameters
        ----------
        source : Source
            The pipeline's data source.
        """
        self._source = source

    def __call__(self, items: Iterator[xr.DataArray], index: int) -> list[str]:
        """Consume DataArrays and write to an HDF5 file.

        Parameters
        ----------
        items : Iterator[xr.DataArray]
            Stream of DataArray items.
        index : int
            Source index.

        Returns
        -------
        list[str]
            Paths of written files.
        """
        relpath = ""
        stem = ""
        if self._source is not None and hasattr(self._source, "relative_path"):
            from pathlib import PurePosixPath

            rel = self._source.relative_path(index)  # ty: ignore[call-non-callable]
            rel_path = PurePosixPath(rel)
            stem = rel_path.stem
            relpath = str(rel_path.parent) if str(rel_path.parent) != "." else ""

        subdir = self._naming_template.format(index=index, relpath=relpath, stem=stem)
        out_path = self._output_dir / subdir / "data.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        import h5py

        paths: list[str] = []
        with h5py.File(str(out_path), "w") as f:
            for da in items:
                f.create_dataset("data", data=da.values)
                paths.append(str(out_path))

        return paths


# Extended API: partition_indices()


class ChunkedZarrSink(Sink["xr.DataArray"]):
    """Example sink demonstrating partition_indices for chunk-aligned writes.

    Parameters
    ----------
    output_path : str
        Path to the output Zarr store.
    chunk_size : int
        Number of indices per Zarr chunk along the time axis.
    """

    name: ClassVar[str] = "Chunked Zarr (Example)"
    description: ClassVar[str] = "Zarr writer with partition constraints"

    @classmethod
    def params(cls) -> list[Param]:
        """Return parameter descriptors.

        Returns
        -------
        list[Param]
            Parameters: output_path and chunk_size.
        """
        return [
            Param(name="output_path", description="Zarr store path", type=str),
            Param(name="chunk_size", description="Indices per chunk", type=int, default=10),
        ]

    def __init__(self, output_path: str, chunk_size: int = 10) -> None:
        self._output_path = output_path
        self._chunk_size = chunk_size

    def __call__(self, items: Iterator[xr.DataArray], index: int) -> list[str]:
        """Write DataArray to the appropriate Zarr chunk.

        Parameters
        ----------
        items : Iterator[xr.DataArray]
            Stream of DataArray items.
        index : int
            Source index.

        Returns
        -------
        list[str]
            Output paths.
        """
        msg = "This is a demonstration stub"
        raise NotImplementedError(msg)

    def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
        """Group indices by Zarr chunk.

        Parameters
        ----------
        indices : list[int]
            The indices to be processed.

        Returns
        -------
        list[list[int]] or None
            Groups of indices (one per chunk), or None if all indices
            fall in the same chunk.
        """
        from collections import defaultdict

        chunk_groups: dict[int, list[int]] = defaultdict(list)
        for idx in indices:
            chunk_id = idx // self._chunk_size
            chunk_groups[chunk_id].append(idx)

        if len(chunk_groups) <= 1:
            return None

        return [sorted(group) for group in chunk_groups.values()]


# Test partition_indices
chunked = ChunkedZarrSink(output_path="output/zarr", chunk_size=10)
groups = chunked.partition_indices(list(range(25)))
print(f"\nChunk partition groups: {groups}")
