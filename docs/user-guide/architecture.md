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

## Design Pattern: Pipes and Filters

PhysicsNeMo Curator implements the **Pipes and Filters** architectural style.
In this pattern a system is decomposed into a series of independent processing
elements (*filters*) connected by channels (*pipes*) that carry a uniform data
stream.  Each filter reads from its input, transforms the data, and writes to
its output without knowledge of its neighbours.

```{note}
**Classical references**

- **Shaw, M. & Garlan, D. (1996).** *Software Architecture: Perspectives on an
  Emerging Discipline.* Prentice Hall, Chapter 2.  Formalises Pipes and Filters
  as one of the canonical architectural styles alongside Layered, Repository,
  and Implicit Invocation.

- **Buschmann, F., Meunier, R., Rohnert, H., Sommerlad, P., & Stal, M.
  (1996).** *Pattern-Oriented Software Architecture, Volume 1: A System of
  Patterns (POSA).* Wiley, pp. 53–70.  Catalogues Pipes and Filters as a
  fundamental architectural pattern with detailed structure, dynamics, and
  known-uses sections.

- **Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994).** *Design
  Patterns: Elements of Reusable Object-Oriented Software.* Addison-Wesley.
  Defines the complementary **Strategy** and **Decorator** patterns used in the
  execution and profiling layers.

- **Hohpe, G. & Woolf, B. (2003).** *Enterprise Integration Patterns.*
  Addison-Wesley, Chapter 3.  Extends Pipes and Filters to message-based and
  streaming integration contexts, directly analogous to the generator-based
  streaming used here.
```

### Mapping to the classical vocabulary

The table below shows how the abstract Pipes and Filters vocabulary maps to
concrete PhysicsNeMo Curator classes.

| Pattern concept | Curator equivalent | Role |
|-----------------|--------------------|------|
| **Data source** | {class}`~physicsnemo_curator.core.base.Source` | Produces typed items from storage |
| **Filter** | {class}`~physicsnemo_curator.core.base.Filter` | Transforms or observes the stream |
| **Pipe** | Python generators (`Generator[T]`) | Lazy connectors between stages |
| **Data sink** | {class}`~physicsnemo_curator.core.base.Sink` | Consumes and persists output |
| **Pipeline** | {class}`~physicsnemo_curator.core.base.Pipeline` | Chains source, filters, sink |

### Why Pipes and Filters?

This pattern is a natural fit for ETL (Extract-Transform-Load) workloads for
several reasons:

1. **Independent stages** — each Source, Filter, and Sink can be developed,
   tested, and reused in isolation.  Any Filter that operates on type `T`
   composes with any Source or Sink of the same type.
2. **Lazy evaluation** — Python generators serve as pipes, so items flow
   through the entire chain one at a time without materialising the full
   dataset in memory.
3. **Parallelism** — because each source index produces an independent stream,
   the pipeline is *embarrassingly parallel* across indices.  The
   {func}`~physicsnemo_curator.run.run_pipeline` function exploits this with six
   pluggable backends (see {doc}`/user-guide/parallel`).
4. **Extensibility** — adding a new stage requires implementing a single
   abstract method (`__getitem__` for sources, `__call__` for filters and
   sinks`) and nothing else.

### Complementary patterns

Beyond the primary Pipes and Filters architecture several classical design
patterns (Gamma et al., 1994) reinforce the framework:

| Pattern | Where used | Purpose |
|---------|-----------|---------|
| **Strategy** | `RunBackend` and its six variants | Decouple definition from execution |
| **Built-in Metrics** | `Pipeline.track_metrics` | Unified checkpointing and profiling |
| **Plugin** | `Registry` + domain submodules | Domain-agnostic core; register at import |
| **Protocol** | `Source` ABC | Decouple discovery from reading |

## Core Components

### Source

{class}`~physicsnemo_curator.core.base.Source` is an abstract base class representing
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

{class}`~physicsnemo_curator.core.base.Filter` transforms a stream of items.  A filter
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

{class}`~physicsnemo_curator.core.base.Sink` persists items and returns file paths:

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

{class}`~physicsnemo_curator.core.base.Pipeline` chains a source through filters into
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

For processing all indices, use {func}`~physicsnemo_curator.core.parallel.run_pipeline`
instead of a manual loop:

```python
from physicsnemo_curator import run_pipeline

# Sequential with progress display
results = run_pipeline(pipeline)

# Parallel across all CPUs
results = run_pipeline(pipeline, n_jobs=-1, backend="process_pool")
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

{class}`~physicsnemo_curator.core.base.Param` describes a configurable parameter on
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
| `da` | `xarray.DataArray` | `da` | Implemented |
| `atm` | `nvalchemi.data.AtomicData` | `atm` | Implemented |

Submodules register their components with the global
{class}`~physicsnemo_curator.core.registry.Registry` at import time, enabling the
CLI to discover them dynamically.

## Registry

The {class}`~physicsnemo_curator.core.registry.Registry` is a global singleton that
tracks all submodules and their pipeline components:

```python
from physicsnemo_curator.core.registry import registry

# Registration happens at import time in each submodule's __init__.py
registry.register_submodule("mesh", "Mesh processing", "physicsnemo.mesh")
registry.register_source("mesh", VTKSource)
registry.register_filter("mesh", MeanFilter)
registry.register_sink("mesh", MeshSink)

# Query
registry.submodules()         # {"mesh": SubmoduleEntry(...)}
registry.sources("mesh")      # {"VTK Reader": <class VTKSource>}
registry.filters("mesh")      # {"Mean Statistics": <class MeanFilter>}
registry.sinks("mesh")        # {"PhysicsNeMo Mesh Writer": <class MeshSink>}
```

Each {class}`~physicsnemo_curator.core.registry.SubmoduleEntry` can check whether its
dependencies are available via the `.available` property.
