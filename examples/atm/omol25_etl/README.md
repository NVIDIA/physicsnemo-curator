# OMol25 Atomic Data ETL

This example demonstrates a complete Source → Filter → Sink pipeline for curating atomic/molecular
data from the Open Molecules 2025 (OMol25) dataset. OMol25 contains over 100 million DFT
calculations covering 83 elements. The pipeline reads raw ASE LMDB files, computes per-field
statistics using numerically stable Welford accumulators, and writes the processed structures to a
Zarr store in the nvalchemi format.

## Prerequisites

```bash
pip install physicsnemo-curator[atm]
```

## Usage

```bash
python main.py
```
