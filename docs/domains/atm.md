<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Atomic Data Submodule

The `curator.atm` submodule provides pipeline components for working with
{class}`~nvalchemi.data.AtomicData` objects — the core data structure in
the [nvalchemi toolkit](https://nvidia.github.io/nvalchemi-toolkit/) for
representing atomic and molecular systems as graphs.

## Installation

```bash
# Install with the atm dependency group
uv sync --group atm
```

Required packages: ``nvalchemi``, ``ase>=3.26.0``, ``torch``.

## Components

### ASELMDBSource

{class}`~physicsnemo_curator.domains.atm.sources.aselmdb.ASELMDBSource` reads
`.aselmdb` database files and yields {class}`~nvalchemi.data.AtomicData` instances.

Each pipeline index corresponds to one `.aselmdb` file.  The generator
iterates over every row in that database, converting each
{class}`~ase.Atoms` entry to {class}`~nvalchemi.data.AtomicData` via
{meth}`AtomicData.from_atoms`.  This 1→N pattern means a directory of 80
files yields 80 indices, each producing thousands of atomic structures.

```python
from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

source = ASELMDBSource(data_dir="./val/")
print(f"{len(source)} database files")  # 80

# Iterate over structures from the first file
for atomic_data in source[0]:
    print(atomic_data.atomic_numbers.shape)
    break
```

The source auto-detects a ``metadata.npz`` file in the data directory if
present.  This file may contain auxiliary arrays such as atom counts per
simulation (``natoms``) and data identifiers (``data_ids``).

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str` | *required* | Directory containing `.aselmdb` files |
| `metadata_path` | `str` | `""` | Path to `metadata.npz` (empty = auto-detect in `data_dir`) |

**Compatible datasets:**

| Dataset | Description | License |
|---------|-------------|---------|
| [OMol25](https://huggingface.co/facebook/OMol25) | 100M+ DFT calculations, 83 elements, ~83M molecular systems | CC-BY-4.0 |
| [OPoly26](https://arxiv.org/abs/2512.23117) | Polymer extension of OMol25 | CC-BY-4.0 |

**References:**

- Levine et al., "The Open Molecules 2025 (OMol25) Dataset, Evaluations,
  and Models", [arXiv:2505.08762](https://arxiv.org/abs/2505.08762) (2025).
- Levine et al., "The Open Polymers 2026 (OPoly26) Dataset and Evaluations",
  [arXiv:2512.23117](https://arxiv.org/abs/2512.23117) (2025).

### AtomicStatsFilter

{class}`~physicsnemo_curator.domains.atm.filters.stats.AtomicStatsFilter` computes
comprehensive per-field statistics for every tensor field in an
{class}`~nvalchemi.data.AtomicData` object.  It is a **pass-through** filter
— items are yielded unchanged for downstream consumption.

Statistics are accumulated internally and written to a Parquet file when
{meth}`flush` is called.  The output includes both human-readable summary
statistics (mean, std, min, max, skewness, kurtosis, etc.) and Welford
accumulator state for exact cross-worker aggregation.

```python
from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter

stats = AtomicStatsFilter(output="stats.parquet")
pipeline = source.filter(stats).write(sink)

for i in range(len(pipeline)):
    pipeline[i]

stats.flush()  # write accumulated statistics
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output` | `str` | *required* | File path for the output Parquet file |

**Computed statistics per field/component:**

| Statistic | Description |
|-----------|-------------|
| `mean` | Arithmetic mean |
| `std` | Population standard deviation |
| `var` | Population variance |
| `min` / `max` | Extremes |
| `median` | Median value |
| `abs_mean` / `abs_max` | Mean and max of absolute values |
| `skewness` | Population skewness |
| `kurtosis` | Excess kurtosis |
| `welford_*` | Internal accumulator state for merging |

**Fields and levels:**

The filter automatically discovers tensor fields on the AtomicData object
and classifies them by semantic level:

- **Node-level** (`positions`, `atomic_numbers`, `forces`, `velocities`, …)
- **Edge-level** (`edge_index`, `shifts`, `unit_shifts`, …)
- **System-level** (`energies`, `stresses`, `virials`, `dipoles`, …)

Vector fields (e.g. `positions` with shape `[n, 3]`) produce one statistics
row per component.  Higher-rank tensors (e.g. `stresses` with shape
`[B, 3, 3]`) are flattened to `[B, 9]` components.

**Parallel merging:**

When running with multiple workers, each worker writes a shard Parquet file.
The static method {meth}`AtomicStatsFilter.merge` (and the public function
{func}`~physicsnemo_curator.domains.atm.filters.stats.merge_welford_stats`) combine
shards using Chan's parallel Welford algorithm — producing exact aggregate
statistics without re-reading raw data.

### AtomicDataZarrSink

{class}`~physicsnemo_curator.domains.atm.sinks.zarr_writer.AtomicDataZarrSink`
writes {class}`~nvalchemi.data.AtomicData` objects to a structured Zarr
store using
{class}`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter`.

Items are collected into configurable batches before being flushed to disk.
The first batch creates the store; all subsequent batches (including those
from different pipeline indices) **append** to the same store, producing a
single consolidated output.

```python
from physicsnemo_curator.domains.atm.sinks.zarr_writer import AtomicDataZarrSink

sink = AtomicDataZarrSink(
    output_path="output.zarr",
    batch_size=1000,  # flush every 1000 items
)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `str` | *required* | Path for the output Zarr store directory |
| `batch_size` | `int` | `1000` | Items per write batch (larger = fewer I/O calls) |

**Output layout** (produced by nvalchemi):

```text
output.zarr/
├── meta/          # atoms_ptr, edges_ptr, samples_mask, ...
├── core/          # atomic_numbers, positions, forces, energies, ...
├── custom/        # user-defined arrays (if any)
└── .zattrs        # root metadata (num_samples, field info)
```

The ``meta/`` group stores CSR-style pointer arrays that enable efficient
random access to individual structures.  The ``core/`` group stores
concatenated per-atom and per-system arrays.

## Full Pipeline Example

```python
from physicsnemo_curator import run_pipeline
from physicsnemo_curator.domains.atm.filters.stats import AtomicStatsFilter
from physicsnemo_curator.domains.atm.sinks.zarr_writer import AtomicDataZarrSink
from physicsnemo_curator.domains.atm.sources.aselmdb import ASELMDBSource

# 1. Source — read .aselmdb files from a local directory
source = ASELMDBSource(data_dir="./val/")
print(f"Database files: {len(source)}")  # 80

# 2. Filter — compute per-field statistics (pass-through)
stats = AtomicStatsFilter(output="outputs/stats.parquet")

# 3. Sink — write to a single Zarr store with batched I/O
sink = AtomicDataZarrSink(
    output_path="outputs/atomic_data.zarr",
    batch_size=1000,
)

# 4. Build pipeline: Source → MeshStatsFilter → Sink
pipeline = source.filter(stats).write(sink)

# 5. Process first 3 files sequentially
results = run_pipeline(pipeline, indices=range(3))

# 6. Flush accumulated statistics
stats.flush()

print(f"Processed {len(results)} database files")
for paths in results:
    print(f"  Written to: {paths}")
```

After running, the outputs are:

```text
outputs/
├── stats.parquet              # per-field statistics with Welford state
└── atomic_data.zarr/
    ├── meta/          # pointer arrays for random access
    ├── core/          # atomic_numbers, positions, forces, energies, ...
    └── .zattrs        # metadata (num_samples, fields)
```

The store can be read back using
{class}`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrReader`:

```python
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader

reader = AtomicDataZarrReader("outputs/atomic_data.zarr")
sample = reader[0]  # AtomicData for the first structure
```

## Data Flow

Each `.aselmdb` file may contain thousands of atomic structures.  The
pipeline streams structures lazily through the generator, batches them in
the sink, and flushes to the Zarr store — so memory usage stays bounded
regardless of file size.

```text
data0000.aselmdb ──[ASELMDBSource]──► AtomicData, ... ──┐
data0001.aselmdb ──[ASELMDBSource]──► AtomicData, ... ──┤
   ...                                                   ├──[AtomicStatsFilter]──►──[AtomicDataZarrSink]──► output.zarr
data0079.aselmdb ──[ASELMDBSource]──► AtomicData, ... ──┘                   │
                                                                            └──► stats.parquet
```

## Dependencies

The `atm` domain depends on:

| Package | Purpose |
|---------|---------|
| [nvalchemi](https://nvidia.github.io/nvalchemi-toolkit/) | `AtomicData` model and Zarr I/O backends |
| [ase](https://wiki.fysik.dtu.dk/ase/) | Atomic Simulation Environment (Atoms objects) |
| [torch](https://pytorch.org/) | Tensor operations (required by nvalchemi) |
