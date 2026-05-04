# DrivAerML End-to-End ETL

This example demonstrates a complete Source → Filter → Sink pipeline that reads DrivAerML meshes
from HuggingFace Hub. The pipeline computes per-field statistics with Welford accumulators, casts
fields to float32, randomly permutes point/cell ordering, and writes outputs grouped into per-run
subdirectories matching the canonical DrivAerML structure.

## Prerequisites

```bash
pip install physicsnemo-curator[mesh]
```

## Usage

```bash
python main.py
```
