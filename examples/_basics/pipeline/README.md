# Creating a Pipeline

This example shows how to build a data curation pipeline using the Source → Filter → Sink pattern.
We read meshes from the Navier-Stokes Cylinder dataset, apply a statistics filter, and write the
outputs to disk. A pipeline is lazy — nothing is executed until you index into it with
`pipeline[i]`.

## Prerequisites

```bash
pip install physicsnemo-curator[mesh]
```

## Usage

```bash
python main.py
```
