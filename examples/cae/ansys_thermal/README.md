# Ansys Thermal Simulation ETL

This example demonstrates a Source → Filter → Sink pipeline for curating Ansys thermal simulation
data. Ansys solvers produce `.rst` result files containing mesh coordinates, temperature
distributions, heat flux vectors, and other physics fields. The pipeline reads these files, logs
mesh metadata, computes summary statistics, converts fields to single precision, and writes the
processed meshes to disk.

## Prerequisites

```bash
pip install physicsnemo-curator[mesh] ansys-dpf-core
```

## Usage

```bash
python main.py
```
