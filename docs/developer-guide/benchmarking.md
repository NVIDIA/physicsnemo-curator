<!--
SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmarking

PhysicsNeMo Curator uses a three-tier benchmarking strategy:

| Tool | Purpose | Scope |
|---|---|---|
| **pytest-benchmark** | Fast per-PR regression checks in CI | Current commit only |
| **ASV (airspeed velocity)** | Long-term historical performance tracking | Across git history |
| **Criterion** | Rust micro-benchmarks | Rust core library |

## Quick Start

```bash
# Install dev dependencies (includes asv + pytest-benchmark)
make install

# Build the native extension
make develop

# Run pytest-benchmark (fast, current code)
make bench

# Run ASV on the current commit
make asv-run

# Preview the ASV dashboard
make asv-preview
```

## pytest-benchmark (CI Benchmarks)

pytest-benchmark runs inside the normal test suite and is designed for fast
per-PR checks. Benchmark tests live in `test/` and use the `benchmark` fixture
provided by pytest-benchmark.

```python
import pytest


@pytest.mark.benchmark
def test_pipeline_throughput(benchmark):
    """Benchmark pipeline item processing."""
    from physicsnemo_curator.core.base import Pipeline

    # ... setup ...
    benchmark(pipeline.__getitem__, 0)
```

Run benchmarks:

```bash
# Run only benchmarks
uv run pytest test/ --benchmark-only

# Skip benchmarks during normal test runs
uv run pytest test/ --benchmark-skip

# Compare against saved baseline
uv run pytest test/ --benchmark-only --benchmark-compare
```

Results are stored as JSON in `.benchmarks/`.

## ASV (Historical Benchmarks)

[Airspeed Velocity](https://asv.readthedocs.io/) tracks performance across the
project's git history and produces an interactive web dashboard. The project
uses `environment_type: "existing"` — benchmarks run in the current
environment rather than isolated virtualenvs, so you must build the extension
first with `make develop`.

### Benchmark Files

ASV benchmarks live in the `benchmarks/` directory at the project root:

```text
benchmarks/
├── __init__.py
├── _helpers.py          # Shared benchmark utilities
├── asv_build.py         # ASV build configuration
├── bench_atm.py         # Atomic data benchmarks
├── bench_backends.py    # Execution backend benchmarks
├── bench_da.py          # DataArray benchmarks
└── bench_mesh.py        # Mesh pipeline benchmarks
```

### Writing ASV Benchmarks

Benchmarks are plain Python classes/functions with magic name prefixes:

| Prefix | Measures |
|---|---|
| `time_` | Wall-clock execution time |
| `mem_` | Memory footprint of returned object |
| `peakmem_` | Peak resident memory |
| `track_` | Arbitrary numeric value |
| `timeraw_` | Execution time in a fresh subprocess |

```python
class TimePipelineIteration:
    """Benchmark per-item pipeline throughput."""

    params = [10, 100, 1000]
    param_names = ["num_items"]

    def setup(self, num_items):
        """Called before each benchmark (excluded from timing)."""
        self.pipeline = build_pipeline(num_items)

    def time_iterate_all(self, num_items):
        """Time iterating through every item."""
        for i in range(len(self.pipeline)):
            self.pipeline[i]
```

### Running ASV

```bash
# Benchmark the current commit
make asv-run

# Dry-run (quick smoke test, no results saved)
make asv-quick

# Benchmark a range of commits
uv run asv run v0.1.0..HEAD

# Compare two revisions
make asv-compare REF1=main REF2=HEAD

# Find the commit that introduced a regression
uv run asv find v0.1.0..HEAD TimePipelineIteration.time_iterate_all

# Show results for a commit
uv run asv show HEAD
```

### Live Dashboard

The ASV benchmark dashboard is published automatically to GitHub Pages by the
nightly CI workflow. View it at:

> **<https://nvidia.github.io/physicsnemo-curator/benchmarks/>**

The dashboard updates each night with the latest results. You can also trigger
a run manually from the **Actions → Benchmark** tab using `workflow_dispatch`.

To preview locally after running benchmarks:

```bash
make asv-publish   # Build static HTML from .asv/results
make asv-preview   # Serve at http://localhost:8080
```

### Configuration

ASV is configured in `asv.conf.json` at the project root. Key settings:

- **`environment_type`**: `existing` (uses the current Python environment —
  run `make develop` before benchmarking)
- **`benchmark_dir`**: Points to `benchmarks/`
- **`regressions_thresholds`**: 5 % regression triggers a warning
- **All ASV artifacts** (envs, results, HTML) are stored under `.asv/` and
  gitignored

## Criterion (Rust Benchmarks)

Rust micro-benchmarks use [Criterion.rs](https://bheisler.github.io/criterion.rs/book/)
and live in `src/rust/benches/`:

```bash
# Run Rust benchmarks
cargo bench --manifest-path src/rust/Cargo.toml
```

Criterion produces HTML reports in `src/rust/target/criterion/`.

## Make Targets Reference

| Target | Description |
|---|---|
| `make bench` | pytest-benchmark + Criterion (fast, current code) |
| `make asv-run` | ASV benchmark on HEAD (saves results) |
| `make asv-quick` | ASV dry-run (no results saved) |
| `make asv-publish` | Build ASV HTML dashboard |
| `make asv-preview` | Serve ASV dashboard locally |
| `make asv-compare REF1=... REF2=...` | Compare two git revisions |
