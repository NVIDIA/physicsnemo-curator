# Creating a Custom Sink

This example shows how to implement and register a custom Sink. We create an `HDF5Sink` that writes
xarray DataArray fields to HDF5 files — one file per source index, with each variable stored as a
separate dataset.

## Prerequisites

```bash
uv sync --extra da
uv pip install h5py
```

## Usage

```bash
uv run python main.py
```

## How It Works

A sink inherits from `Sink` and implements three things:

1. `name` / `description` class variables
2. `params()` class method
3. `__call__(items, index)` — consume the iterator, write data, return a list of written file paths

### Step 1 — Define the Sink

```python
from physicsnemo_curator.core.base import Param, Sink

class HDF5Sink(Sink["xr.DataArray"]):
    name: ClassVar[str] = "HDF5 Writer"
    description: ClassVar[str] = "Write DataArrays to HDF5 files"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="output_dir", description="Output directory for HDF5 files", type=str),
            Param(name="compression", description="HDF5 compression filter", type=str,
                  default="gzip", choices=["gzip", "lzf"]),
            Param(name="compression_level", description="Compression level (0=off, 9=max)",
                  type=int, default=4),
        ]

    def __init__(self, output_dir: str, compression: str = "gzip", compression_level: int = 4) -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._compression = compression
        self._compression_level = compression_level

    def __call__(self, items: Iterator[xr.DataArray], index: int) -> list[str]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        h5_path = self._output_dir / f"data_{index:04d}.h5"
        # ... write items to HDF5, return [str(h5_path)]
```

### Step 2 — Register the Sink (Optional)

Registration makes the sink discoverable in the global registry:

```python
from physicsnemo_curator.core.registry import registry

registry.register_sink("da", HDF5Sink)
```

### Step 3 — Use in a Pipeline

The custom sink plugs into the standard pipeline API:

```python
source = ERA5Source(
    times=[datetime(2020, 6, 1, 0), datetime(2020, 6, 1, 6)],
    variables=["t2m", "u10m"],
    backend="arco",
)

pipeline = source.write(HDF5Sink(output_dir="output/extending/hdf5/"))

results = run_pipeline(pipeline, n_jobs=1, backend="sequential", indices=range(len(pipeline)), use_tui=True)
```

### Step 4 — Verify Output

Read back the HDF5 file to confirm the data was written correctly:

```python
import h5py

with h5py.File(results[0][0], "r") as f:
    for key in f:
        ds = f[key]
        print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")
```

## Extended API: `set_source()`

Sinks that need source metadata for output naming can implement `set_source()`. The Pipeline
automatically calls it when the sink is attached via `.write()`.

Common use case: resolving naming placeholders like `{relpath}`, `{stem}`, `{run_id}`,
`{mesh_name}` based on source structure.

See `TemplatedHDF5Sink` in `main.py` for a complete implementation.

## Extended API: `partition_indices()`

Sinks with concurrent write constraints override `partition_indices()` to group indices that must
be processed by the same worker.

Common use case: a Zarr sink where multiple indices write to the same chunk — they must go through
the same worker to avoid corruption.

```python
def partition_indices(self, indices: list[int]) -> list[list[int]] | None:
    # Group indices by Zarr chunk
    chunk_groups = defaultdict(list)
    for idx in indices:
        chunk_id = idx // self._chunk_size
        chunk_groups[chunk_id].append(idx)
    return [sorted(group) for group in chunk_groups.values()]
```

When both source and sink provide `partition_indices`, the framework computes the intersection
(finest partition satisfying both constraints) using `intersect_partitions()`, then assigns groups
to workers with `batch_groups()`.

## Summary

To create a custom sink:

1. Subclass `Sink` with a type parameter (`Sink["xr.DataArray"]`, `Sink["Mesh"]`, etc.)
2. Set `name` and `description` class variables
3. Implement `params()` and `__call__(items, index) -> list[str]`
4. Ensure the output directory is created automatically
5. Return `[]` for empty iterators (no crash, no empty files)
6. Optionally register with `registry.register_sink()`

Extended API:

- `set_source(source)` — receive source reference for naming placeholders (called automatically
  by `Pipeline.write()` if the method exists)
- `partition_indices(indices) -> list[list[int]] | None` — group indices that must share a worker
  (e.g. same Zarr chunk, same output file)
