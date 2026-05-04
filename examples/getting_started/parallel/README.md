# Running a Pipeline in Parallel

This example demonstrates `run_pipeline` to execute a pipeline across multiple source indices using
parallel workers. Building on the Creating a Pipeline example, we process multiple DrivAerML CFD
meshes concurrently with a `process_pool` backend and then merge per-worker statistics using
`gather_pipeline`.

## Prerequisites

```bash
pip install physicsnemo-curator[mesh]
```

## Usage

```bash
python main.py
```
