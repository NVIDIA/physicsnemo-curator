# Benchmarks

PhysicsNeMo Curator tracks performance across commits using
[ASV (Airspeed Velocity)](https://asv.readthedocs.io/).  The dashboard
compares Rust and Python backends for I/O readers, pipeline throughput,
and memory usage.

```{raw} html
<p><a href="benchmarks/index.html" class="btn btn-primary">Open ASV Dashboard</a></p>
```

The benchmark suite runs nightly and on every push to `main`/`refactor`.
See the [developer guide](developer-guide/benchmarking.md) for instructions
on running benchmarks locally.
