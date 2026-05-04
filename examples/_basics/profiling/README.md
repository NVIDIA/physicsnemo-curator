# Profiling a Pipeline

This example demonstrates how to measure pipeline performance using `ProfiledPipeline`. It is a
transparent wrapper that records per-index and per-stage wall-clock timing without changing the
pipeline's behaviour. It can be passed directly to `run_pipeline` — backends see it as a regular
pipeline.

## Prerequisites

```bash
pip install physicsnemo-curator[mesh]
```

## Usage

```bash
python main.py
```
