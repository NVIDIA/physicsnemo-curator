# Checkpointing a Pipeline

This example demonstrates how to checkpoint pipeline execution using `CheckpointedPipeline`. It
wraps a pipeline and records completed indices in a SQLite database. If the pipeline is interrupted
and restarted, already-completed indices are skipped and their cached output paths are returned
immediately. This is especially useful for long-running pipelines over large datasets where you want
crash resilience without re-processing.

## Prerequisites

```bash
pip install physicsnemo-curator[mesh]
```

## Usage

```bash
python main.py
```
