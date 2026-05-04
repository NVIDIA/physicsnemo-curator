# Crash Simulation ETL

This example demonstrates a Source → Filter → Sink pipeline for curating LS-DYNA crash simulation data.
Automotive crash simulations produce multi-timestep shell meshes stored in the `d3plot` binary format.
The pipeline reads these files, removes non-deforming wall nodes, logs mesh metadata, converts fields
to single precision, and writes the processed meshes to disk.

## Prerequisites

```bash
pip install physicsnemo-curator[mesh] lasso-python
```

## Usage

```bash
python main.py
```
