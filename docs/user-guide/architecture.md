# Architecture

PhysicsNeMo Curator is built around three core abstractions — **Source**,
**Filter**, and **Sink** — that compose into lazy **Pipelines**.

## Overview

```text
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Source   │────▶│ Filter A │────▶│ Filter B │────▶│   Sink   │
│ (reader)  │     │(transform)│    │(transform)│    │ (writer) │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
  yields T        T → T            T → T           T → [paths]
```

Within a submodule (e.g. `mesh`), all components communicate through a
single data type `T`.  For the mesh submodule, `T` is
{class}`physicsnemo.mesh.Mesh`.

## Core Components

### Source

{class}`~curator.core.base.Source` is an abstract base class representing
a collection of data items.  Sources are indexed by integer and yield
items as generators:

```python
class Source[T](ABC):
    name: ClassVar[str]
    description: ClassVar[str]

    @classmethod
    def params(cls) -> list[Param]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Generator[T]: ...
```

Key properties:

- **Sized** — `len(source)` returns the number of available items
- **Generator semantics** — `source[i]` returns a generator that may yield
  one or more items (e.g. a multi-block VTK file could yield multiple meshes)
- **Builder methods** — `.filter(f)` and `.write(s)` create pipelines

### Filter

{class}`~curator.core.base.Filter` transforms a stream of items.  A filter
is a callable that receives a generator and returns a generator:

```python
class Filter[T](ABC):
    name: ClassVar[str]
    description: ClassVar[str]

    @classmethod
    def params(cls) -> list[Param]: ...
    def __call__(self, items: Generator[T]) -> Generator[T]: ...
```

Filters have full generator semantics:

| Behaviour | Description | Example |
|-----------|-------------|---------|
| **Pass-through** | Yield every item unchanged | `MeanFilter` (computes stats as side effect) |
| **Transform** | Yield one modified item per input | Scaling, normalization |
| **Expand** | Yield multiple items per input | Mesh subdivision, augmentation |
| **Contract** | Skip some items | Quality filtering, deduplication |

Stateful filters (like `MeanFilter`) may accumulate data across items and
expose a `flush()` method to finalize their output.

### Sink

{class}`~curator.core.base.Sink` persists items and returns file paths:

```python
class Sink[T](ABC):
    name: ClassVar[str]
    description: ClassVar[str]

    @classmethod
    def params(cls) -> list[Param]: ...
    def __call__(self, items: Generator[T], index: int) -> list[str]: ...
```

The sink receives both the item stream and the source index (for naming
output files).

### Pipeline

{class}`~curator.core.base.Pipeline` chains a source through filters into
a sink.  Pipelines are **immutable** — `.filter()` and `.write()` return
new instances:

```python
pipeline = (
    Source(...)
    .filter(FilterA())
    .filter(FilterB())
    .write(Sink(...))
)

# Lazy per-item execution
paths = pipeline[i]   # returns list[str]
len(pipeline)          # delegates to len(source)
```

Execution flow for `pipeline[i]`:

1. Call `source[i]` to get a generator of `T`
2. Chain through each filter: `stream = filter(stream)`
3. Feed into the sink: `sink(stream, index=i)`
4. Return the list of output file paths

### Batch & Parallel Execution

For processing all indices, use {func}`~curator.core.parallel.run_pipeline`
instead of a manual loop:

```python
from curator import run_pipeline

# Sequential with progress bar
results = run_pipeline(pipeline)

# Parallel across all CPUs
results = run_pipeline(pipeline, n_jobs=-1, backend="processes")
```

`run_pipeline` supports multiple backends — `"sequential"`, `"processes"`,
`"loky"` (joblib), and `"dask"` — with automatic detection of the best
available option.  See {doc}`/user-guide/parallel` for details.

```{important}
Multiprocess backends execute each index in a **separate process** with
an independent copy of the pipeline.  Stateful filters (e.g.
``MeanFilter._rows``) accumulate per-process state that is not merged
back.  Use sequential execution when filter side-effects must be
aggregated.
```

### Param

{class}`~curator.core.base.Param` describes a configurable parameter on
any component.  It drives the interactive CLI prompts:

```python
@dataclass(frozen=True)
class Param:
    name: str                     # matches __init__ kwarg
    description: str              # help text for CLI
    type: type = str              # expected Python type
    default: Any = REQUIRED       # sentinel = must be provided
    choices: list[str] | None = None  # restrict to specific values
```

## Submodules

Each domain vertical is a submodule with its own data type and dependency
group:

| Submodule | Data Type | Dependency Group | Status |
|-----------|-----------|-----------------|--------|
| `mesh` | `physicsnemo.mesh.Mesh` | `mesh` | Implemented |
| `xr` | `xarray.Dataset` | `xr` | Planned |
| `mdt` | `tuple[torch.Tensor, ...]` | `mdt` | Planned |

Submodules register their components with the global
{class}`~curator.core.registry.Registry` at import time, enabling the
CLI to discover them dynamically.

## FileStore

{class}`~curator.core.store.FileStore` is a protocol that decouples
**file discovery and access** from **file reading**.  Sources accept a
`FileStore` via dependency injection, so the same reader works with local
directories, S3 buckets, HuggingFace Hub datasets, or any custom backend.

```python
@runtime_checkable
class FileStore(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> str: ...
    # returns a local filesystem path
```

Built-in implementations are registered with each submodule's
{class}`~curator.core.registry.Registry` and are selectable in the CLI.
Users can also register custom stores at runtime (see {ref}`store-registration`
above).

Two built-in implementations are provided:

### LocalFileStore

{class}`~curator.core.store.LocalFileStore` discovers and serves files
from a local directory using `pathlib.Path.glob`:

```python
from curator.core.store import LocalFileStore

store = LocalFileStore("./data/", extensions=frozenset({".vtk", ".vtu"}))
path = store[0]  # "/absolute/path/to/data/mesh_0000.vtu"
```

### FsspecFileStore

{class}`~curator.core.store.FsspecFileStore` discovers and serves files
from any `fsspec`-compatible URL, transparently caching downloads:

```python
from curator.core.store import FsspecFileStore

# HuggingFace Hub
store = FsspecFileStore(
    "hf://datasets/neashton/drivaerml/run_1/slices",
    extensions=frozenset({".vtk", ".vtp"}),
)

# S3 (public bucket)
store = FsspecFileStore(
    "s3://my-bucket/cfd-data/",
    storage_options={"anon": True},
)

path = store[0]  # local cached path, ready for pyvista.read()
```

Remote files are downloaded once and cached locally.  The cache location
can be controlled via the `cache_storage` parameter.

### Custom stores

Any object implementing `__len__` and `__getitem__` (returning a local
path string) satisfies the `FileStore` protocol:

```python
class DatabaseFileStore:
    """Fetch VTK files from a database by row id."""
    def __len__(self) -> int:
        return self._db.count()
    def __getitem__(self, index: int) -> str:
        return self._db.download_to_cache(index)
```

## Registry

The {class}`~curator.core.registry.Registry` is a global singleton that
tracks all submodules, their pipeline components, and their file stores:

```python
from curator.core.registry import registry

# Registration happens at import time in each submodule's __init__.py
registry.register_submodule("mesh", "Mesh processing", "physicsnemo.mesh")
registry.register_store("mesh", "Local directory", LocalFileStore)
registry.register_store("mesh", "Remote (fsspec)", FsspecFileStore)
registry.register_source("mesh", VTKSource)
registry.register_filter("mesh", MeanFilter)
registry.register_sink("mesh", MeshSink)

# Query
registry.submodules()         # {"mesh": SubmoduleEntry(...)}
registry.stores("mesh")       # {"Local directory": <class LocalFileStore>, ...}
registry.sources("mesh")      # {"VTK Reader": <class VTKSource>}
registry.filters("mesh")      # {"Mean Statistics": <class MeanFilter>}
registry.sinks("mesh")        # {"PhysicsNeMo Mesh Writer": <class MeshSink>}
```

Each {class}`~curator.core.registry.SubmoduleEntry` can check whether its
dependencies are available via the `.available` property.

(store-registration)=

### Store Registration

File stores are registered per-submodule with a human-readable display
name.  The built-in stores (`LocalFileStore` and `FsspecFileStore`) are
registered automatically when a submodule is imported.  Users can register
custom stores at runtime:

```python
from curator.core.registry import registry

registry.register_store("mesh", "My Database Store", DatabaseFileStore)
```

The interactive CLI uses the store registry to present data-source options
before prompting for a reader.
