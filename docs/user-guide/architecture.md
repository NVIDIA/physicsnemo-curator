# Architecture

PhysicsNeMo Curator is built around three core abstractions — **Source**,
**Filter**, and **Sink** — that compose into lazy **Pipelines**.

## Overview

```
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

## Registry

The {class}`~curator.core.registry.Registry` is a global singleton that
tracks all submodules and their components:

```python
from curator.core.registry import registry

# Registration happens at import time in each submodule's __init__.py
registry.register_submodule("mesh", "Mesh processing", "physicsnemo.mesh")
registry.register_source("mesh", VTKSource)
registry.register_filter("mesh", MeanFilter)
registry.register_sink("mesh", MeshSink)

# Query
registry.submodules()         # {"mesh": SubmoduleEntry(...)}
registry.sources("mesh")      # {"VTK Reader": <class VTKSource>}
```

Each {class}`~curator.core.registry.SubmoduleEntry` can check whether its
dependencies are available via the `.available` property.
