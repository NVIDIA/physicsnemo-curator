# OMol25 Atomic Data ETL

Process [OMol25][omol25] atomic/molecular DFT data through a complete
Source → Filter → Sink pipeline.

The pipeline:

1. **AtomicStatsFilter** — computes per-field, per-component statistics
   (mean, std, skewness, kurtosis) using Welford accumulators that merge
   across parallel workers.
2. **AtomicDataZarrSink** — batches structures and writes them to a Zarr
   store in the nvalchemi AtomicData layout (CSR-style pointer arrays
   for variable-size systems).

## Prerequisites

```bash
uv sync --group mesh

# or with pip
pip install physicsnemo-curator[atm]
pip install huggingface_hub[cli]
```

## Download the Dataset

OMol25 is hosted on HuggingFace at [facebook/OMol25][omol25-data].
The repository is **gated** — you must accept the license terms on the
HuggingFace page before downloading.

> **Warning:** The full OMol25 dataset is very large (100M+ DFT
> calculations across ~80 `.aselmdb` files). For this example only the
> **validation split** is needed. Ensure you have sufficient disk space
> before downloading.

```bash
# Authenticate with HuggingFace (required for gated repos)
huggingface-cli login

# Download the validation split tar.gz archive
huggingface-cli download facebook/OMol25 \
    --include "val.tar.gz" \
    --local-dir input/omol25

# Extract the .aselmdb files
tar -xzf input/omol25/val.tar.gz -C input/omol25
```

This creates an `input/omol25/val/` directory with LMDB database shards
and a metadata file:

```text
input/omol25/val/
├── 0000.aselmdb           # LMDB database files (one per shard)
├── 0001.aselmdb
├── ...
└── metadata.npz           # Metadata (natoms, data_ids arrays)
```

## Usage

```bash
# Basic usage (reads from ./input/omol25/val, writes to ./output/omol25)
python main.py

# Custom input/output directories
python main.py --input /path/to/omol25/val --output /path/to/output

# Process more files with more workers
python main.py --n-indices 10 --workers 4
```

## Output Structure

```text
output/omol25/
├── stats.parquet              # Per-field statistics (merged)
└── dataset.zarr/              # AtomicData Zarr store
    ├── meta/                  # Pointer arrays (atoms_ptr, edges_ptr)
    ├── core/                  # Core fields (positions, forces, ...)
    ├── custom/                # User-defined fields
    └── .zattrs                # Root metadata (num_samples, fields)
```

[omol25]: https://huggingface.co/facebook/OMol25
[omol25-data]: https://huggingface.co/facebook/OMol25/blob/main/DATASET.md
