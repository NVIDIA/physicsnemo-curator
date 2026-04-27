# Architecture

PhysicsNeMo Curator is a lazy-evaluation ETL pipeline system for curating physics
simulation data. It provides a Python API backed by a Rust native extension for
performance-critical parsing, with tooling for interactive pipeline construction,
parallel execution, caching, and metrics visualization.

## Package Layout

```
src/physicsnemo_curator/
├── __init__.py           # Public API: Source, Filter, Sink, Pipeline, Param, run_pipeline
├── _lib.pyi              # Type stubs for Rust extension
├── core/                 # Pipeline framework
│   ├── base.py           # Abstract base classes (Source, Filter, Sink, Pipeline, Param)
│   ├── pipeline_store.py # SQLite metrics store and checkpointing
│   ├── registry.py       # Component discovery and registration
│   ├── cache.py          # Cache directory management (XDG-compliant)
│   └── serialization.py  # YAML/JSON pipeline round-trip
├── run/                  # Execution backends
│   ├── base.py           # RunBackend ABC + RunConfig
│   ├── sequential.py     # Single-threaded for-loop
│   ├── thread_pool.py    # ThreadPoolExecutor (I/O-bound)
│   ├── process_pool.py   # ProcessPoolExecutor (CPU-bound)
│   ├── loky.py           # joblib/loky (robust multiprocessing)
│   ├── dask.py           # Distributed execution via dask.bag
│   └── prefect.py        # Workflow orchestration with Prefect
├── domains/              # Domain-specific sources, filters, and sinks
├── dashboard/            # Metrics visualization (Panel web app)
│   ├── app.py            # DashboardApp server
│   ├── data.py           # DashboardStore (SQLite query layer)
│   ├── views/            # Tab views (overview, pipeline, performance)
│   └── widgets/          # Per-filter visualization widgets
└── wiz/                  # Interactive TUI wizard (Textual)
    ├── app.py            # CuratorApp + WizardState
    └── screens/          # Step-by-step pipeline builder screens

src/rust/src/             # Native extension (PyO3)
├── lib.rs                # Module registration
├── d3plot/               # LS-DYNA binary parsing + tensor math
├── vtk/                  # Multi-threaded VTK XML reading
└── lmdb/                 # ASE LMDB deserialization
```

---

## Pipeline Model

The core abstraction is a lazy, composable ETL chain:

```
Source[T] → Filter[T]* → Sink[T]
```

- **Source** — indexed collection of data. `__len__` returns the count,
  `__getitem__(i)` yields a generator of `T` items for index `i`.
- **Filter** — stream transform. Receives `Generator[T]`, yields `Generator[T]`.
  Can expand (one-to-many), contract (many-to-one), or pass through.
- **Sink** — terminal consumer. Receives `Iterator[T]` + index, writes to storage,
  returns the list of output file paths.
- **Pipeline** — immutable builder tying these together. `.filter(f)` and `.write(s)`
  return *new* pipelines (no mutation). Executing `pipeline[i]` triggers the lazy chain.

### Key Design Choices

1. **Lazy evaluation** — nothing executes until `pipeline[index]` or `run_pipeline()`.
2. **Immutable builder** — `pipeline.filter(f)` returns a new instance; the original is unchanged.
3. **Generator semantics** — items stream through stages without full materialization.
4. **Per-index isolation** — each index is self-contained and independently cacheable.

### Param

Every component declares its configuration via `Param` descriptors:

```python
@classmethod
def params(cls) -> list[Param]:
    return [
        Param("input_path", "Directory to read from", type=pathlib.Path),
        Param("threshold", "Quality threshold", type=float, default=0.5),
    ]
```

The wizard, serialization layer, and CLI all consume `Param` metadata to dynamically
generate forms, validate inputs, and reconstruct components from saved configs.

---

## Execution

### Single-Index Execution (`pipeline[i]`)

```
source[i] → gen₀
filter₁(gen₀) → gen₁
filter₂(gen₁) → gen₂
sink(gen₂, i) → ["/out/file_i.ext"]
```

When `track_metrics=True` (the default), each stage is wrapped with a
`_TimedGenerator` that records wall-clock time, `tracemalloc` captures peak memory,
and results are stored in the SQLite pipeline store.

### Parallel Execution (`run_pipeline`)

```python
results = run_pipeline(pipeline, n_jobs=4, backend="process_pool")
```

`run_pipeline` dispatches all source indices to workers via a pluggable backend:

| Backend | When to use |
|---------|-------------|
| `sequential` | Debugging, stateful filters |
| `thread_pool` | I/O-bound (network, disk) |
| `process_pool` | CPU-bound (GIL-free parallelism) |
| `loky` | Like process_pool but handles complex pickling |
| `dask` | Distributed clusters |
| `prefect` | Workflow orchestration + observability |

Backend selection:
- `n_jobs=1` → always sequential
- `backend="auto"` → picks best available (dask > loky > process_pool)

Each worker gets an independent copy of the pipeline (pickled), processes its
assigned indices, and records metrics to the shared SQLite database.

### Stateful Filters & Gather

Some filters accumulate state across items (e.g., running statistics). In parallel
execution, each worker accumulates independently into shard files. After `run_pipeline`
completes, call:

```python
merged_paths = gather_pipeline(pipeline)
```

This discovers shard files, calls the filter's `merge()` method, and removes the shards.

### Custom Backends

```python
class MyBackend(RunBackend):
    name = "my_backend"
    description = "Custom execution"
    requires = ("some-package",)

    def run(self, pipeline, config):
        # Distribute work, return list[list[str]]
        ...

register_backend(MyBackend)
```

---

## Caching & Database

### PipelineStore (SQLite)

Every pipeline execution creates (or reuses) a SQLite database in the cache directory.
The database is identified by a SHA-256 hash of the pipeline's serialized configuration.

**Tables:**
- `pipeline_runs` — one row per invocation (config hash, start time)
- `index_results` — per-index completion status + output paths
- `worker_info` — PID, hostname, thread/process ID per worker
- `stage_timings` — per-stage wall-clock time for each index
- `memory_usage` — peak memory + optional GPU memory per index
- `artifacts` — side-effect files from stateful filters

**Checkpointing:**
Before processing an index, the pipeline checks `is_completed(index)`. If the index
was already processed (by a prior run or another worker), the cached paths are returned
immediately. This makes pipelines **resumable** — kill a run and restart it without
reprocessing completed indices.

### Cache Directory

Follows XDG Base Directory Specification:
1. `$PSNC_CACHE_DIR` (explicit override)
2. `$XDG_CACHE_HOME/psnc/`
3. `~/.cache/psnc/`

The `cache` module provides introspection:
- `list_databases()` → returns `DBInfo` objects with metadata (hash, size, source,
  sink, completion counts)
- Used by the wizard's cache management screen and dashboard

---

## Registry

Components are discovered via a global `Registry` singleton. Each domain registers
its sources, filters, and sinks at import time:

```python
registry.register_submodule("mesh", "Mesh processing", "physicsnemo.mesh")
registry.register_source("mesh", VTKSource)
registry.register_filter("mesh", StatsFilter)
registry.register_sink("mesh", MeshSink)
```

The registry tracks:
- **Submodules** — named groups with an `import_check` module path
- **Availability** — a submodule is "available" only if its dependency can be imported
- **Component classes** — keyed by their `name` class attribute

Consumers (wizard, serialization) call `registry.sources("mesh")` etc. to get the
available components. This plugin architecture means adding a new domain requires only
writing the components and registering them — no changes to core.

---

## Serialization

Pipelines serialize to YAML or JSON for sharing, version control, and the wizard's
load/save functionality:

```yaml
version: 1
metadata:
  psnc_version: "0.1.0"
  rust_extension: "0.1.0"
  python_version: "3.12.3"
  platform: "Linux-6.1.0-x86_64"
  created_utc: "2026-04-27T15:30:00Z"
  git_hash: "a1b2c3d4e5f6..."
  git_dirty: false
source:
  name: VTK
  module: physicsnemo_curator.domains.mesh.sources.vtk
  params:
    input_path: /data/meshes
    file_pattern: "**/*.vtu"
filters:
  - name: MeshQuality
    module: physicsnemo_curator.domains.mesh.filters.quality
    params:
      threshold: 0.5
sink:
  name: MeshSink
  module: physicsnemo_curator.domains.mesh.sinks.mesh_writer
  params:
    output_path: /out
```

The `metadata` block captures provenance (package version, git state, platform,
timestamp) and is purely informational — it is ignored on deserialization.

`load_pipeline()` reconstructs live objects via `importlib` + `Param.type` coercion.
Round-trip fidelity is guaranteed: `load_pipeline(save_pipeline(p))` produces an
equivalent executable pipeline.

---

## Dashboard

A web-based metrics viewer built on [Panel](https://panel.holoviz.org/) with Material
UI theming. It reads from the SQLite pipeline store and renders three tabs:

1. **Overview** — aggregate stats (total indices, elapsed time, peak memory)
2. **Pipeline** — per-index details, stage timing breakdown, filter artifact previews
3. **Performance** — historical trends, worker distribution, bottleneck identification

Filters can provide custom dashboard widgets by overriding `dashboard_panel()` and
`dashboard_layout_hints()`. These are rendered in the Pipeline tab alongside the
standard metrics.

Launch:
```bash
curator dashboard /path/to/pipeline.db
```

---

## Wizard (TUI)

An interactive terminal wizard built on [Textual](https://textual.textualize.io/) that
guides users through pipeline construction without writing code.

**Flow:**
```
Welcome → Submodule → Source → Filters → Sink → Summary → Execute → Results
                                                    ↕
                                              Cache Manager
```

The wizard queries the registry to populate selection lists and dynamically generates
parameter forms from `Param` descriptors. Built pipelines can be saved to YAML/JSON
or executed immediately with a chosen backend.

**State** is managed via a shared `WizardState` dataclass passed through all screens.
Navigation uses Textual's screen stack (`push_screen` / `pop_screen`).

Launch:
```bash
psnc
```

---

## Rust Extension

Performance-critical parsing is implemented in Rust and exposed via PyO3. The native
module is built with maturin and imported as `physicsnemo_curator._lib`.

| Module | Purpose |
|--------|---------|
| `d3plot` | LS-DYNA crash simulation binary parsing, node thickness computation, von Mises stress |
| `vtk` | Multi-threaded VTK XML reading with direct NumPy conversion |
| `lmdb` | ASE LMDB format deserialization (zlib + JSON, bypasses Python stack) |

The Rust extension is **optional** — the package degrades gracefully if not built
(e.g., during docs builds). Domain sources that need Rust call into `_lib` internally.

---

## Domains

Each domain is a self-contained package under `domains/` with the same structure:

```
domain_name/
├── __init__.py     # Registry registration
├── sources/        # Source implementations
├── filters/        # Filter implementations
└── sinks/          # Sink implementations
```

Domains are **isolated** — they only depend on `core` and their own external
libraries. A domain's availability is gated by whether its `import_check` module
can be imported (e.g., `physicsnemo.mesh` for the mesh domain).

---

## Data Flow (End to End)

```
┌─────────────────────────────────────────────────────┐
│  User (Python API / Wizard / CLI)                   │
│  pipeline = Source(...).filter(...).write(Sink(...)) │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼─────────────────┐
         │  run_pipeline(pipeline, ...)   │
         │  → resolve backend            │
         │  → build RunConfig            │
         └─────────────┬─────────────────┘
                       │
         ┌─────────────▼─────────────────────────┐
         │  Backend distributes indices to workers│
         └─────────────┬─────────────────────────┘
                       │
         ┌─────────────▼─────────────────────────┐
         │  Worker: pipeline[index]               │
         │  1. Check cache → return if hit        │
         │  2. source[i] → gen                    │
         │  3. filter chain → gen                 │
         │  4. sink(gen, i) → paths               │
         │  5. Record metrics to SQLite           │
         └─────────────┬─────────────────────────┘
                       │
         ┌─────────────▼─────────────────────────┐
         │  PipelineStore (SQLite)                │
         │  • Checkpoint per-index completion     │
         │  • Stage timings + memory usage        │
         │  • Worker provenance                   │
         └─────────────┬─────────────────────────┘
                       │
         ┌─────────────▼─────────────────────────┐
         │  Post-run (optional)                   │
         │  • gather_pipeline() for shard merge   │
         │  • Dashboard visualization             │
         │  • Metrics export (CSV/JSON/console)   │
         └───────────────────────────────────────┘
```

---

## Concurrency Model

- **Pipeline** holds a `threading.Lock` for lazy `PipelineStore` initialization
  (double-check locking pattern).
- **SQLite** uses WAL mode for concurrent writes from multiple workers.
- **Parallel backends** give each worker an independent (pickled) pipeline copy —
  no shared mutable state between workers.
- **Stateful filters** accumulate per-worker; use `gather_pipeline()` to merge after.

---

## Extension Points

| Want to... | Do this |
|------------|---------|
| Add a data source | Subclass `Source[T]`, implement `params/__len__/__getitem__`, register |
| Add a transform | Subclass `Filter[T]`, implement `params/__call__`, register |
| Add an output format | Subclass `Sink[T]`, implement `params/__call__`, register |
| Add a domain | Create `domains/name/`, register submodule + components |
| Add an execution backend | Subclass `RunBackend`, call `register_backend()` |
| Add a dashboard widget | Override `Filter.dashboard_panel()` on your filter |
| Use from code | `from physicsnemo_curator import Source, Pipeline, run_pipeline` |
| Use interactively | Run `psnc` for the TUI wizard |
