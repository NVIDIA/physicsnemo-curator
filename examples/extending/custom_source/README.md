# Creating a Custom Source

This example shows how to implement and register a custom Source. We create a `CylinderFlowSource`
that reads the Navier-Stokes Cylinder dataset from HuggingFace Hub using Parquet files. This
demonstrates the core source contract: indexed access with generator semantics, lazy loading, and
shared geometry caching.

## Prerequisites

```bash
pip install physicsnemo-curator[mesh]
```

## Usage

```bash
python main.py
```
