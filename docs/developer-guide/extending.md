# Writing Custom Components

This guide explains how to create custom sources, filters, and sinks for
PhysicsNeMo Curator.

## Anatomy of a Component

Every component has:

1. **`name`** and **`description`** class variables for CLI display
2. A **`params()`** classmethod returning a list of {class}`~physicsnemo_curator.core.base.Param`
3. An **`__init__`** accepting those parameters as keyword arguments
4. A core method (`__getitem__`, `__call__`, or `__call__`) implementing the logic

## Custom FileStore

A {class}`~physicsnemo_curator.core.store.FileStore` maps integer indices to local
file paths.  Any object with `__len__` and `__getitem__` (returning a
`str` path) satisfies the protocol:

```python
class DatabaseFileStore:
    """Fetch files from a database by row id."""

    def __init__(self, connection_string: str, table: str) -> None:
        self._db = connect(connection_string)
        self._rows = self._db.query(f"SELECT id FROM {table}")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> str:
        row_id = self._rows[index]
        return self._db.download_to_cache(row_id)
```

Register custom stores with the global registry so the interactive CLI
can offer them as a data-source option:

```python
from physicsnemo_curator.core.registry import registry

registry.register_store("mesh", "My Database", DatabaseFileStore)
```

The CLI will then show "My Database" alongside "Local directory" and
"Remote (fsspec)" in the store selection prompt.

## Custom Source

A source reads data from a {class}`~physicsnemo_curator.core.store.FileStore` and
yields items.  Subclass {class}`~physicsnemo_curator.core.base.Source` and implement
`__len__`, `__getitem__`, and `params`:

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from physicsnemo_curator.core.base import Source, Param
from physicsnemo_curator.core.store import FileStore

if TYPE_CHECKING:
    from collections.abc import Generator
    from physicsnemo.mesh import Mesh

class MySource(Source["Mesh"]):
    name: ClassVar[str] = "My Reader"
    description: ClassVar[str] = "Reads data from my custom format"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="option", description="Processing option", type=str, default="default"),
        ]

    def __init__(self, store: FileStore, option: str = "default") -> None:
        self._store = store
        self._option = option

    def __len__(self) -> int:
        return len(self._store)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        path = self._store[index]
        mesh = self._load(path)
        yield mesh

    def _load(self, path: str) -> Mesh:
        """Load a single item from a local file path."""
        ...
```

Key points:

- The source receives a `FileStore` — it never handles file discovery or
  downloads directly.
- `__getitem__` is a **generator** (uses `yield`).  It can yield multiple
  items per index if needed.
- `params()` drives the CLI prompts and documents the constructor interface.
  Do **not** include a `store` parameter — the CLI constructs the store
  separately.

## Custom Filter

A filter transforms a stream of items.  Subclass
{class}`~physicsnemo_curator.core.base.Filter`:

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator
    from physicsnemo.mesh import Mesh

class ScaleFilter(Filter["Mesh"]):
    name: ClassVar[str] = "Scale"
    description: ClassVar[str] = "Scale mesh points by a constant factor"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="factor", description="Scale factor", type=float),
        ]

    def __init__(self, factor: float) -> None:
        self._factor = factor

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        for mesh in items:
            mesh.points *= self._factor
            yield mesh
```

### Filter Patterns

**Pass-through with side effects** (like `MeanFilter`):

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        self._accumulate(mesh)
        yield mesh  # unchanged
```

**Expand** (yield multiple items per input):

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        yield mesh                    # original
        yield self._augment(mesh)     # augmented copy
```

**Contract** (skip items):

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        if self._passes_quality_check(mesh):
            yield mesh
```

### Stateful Filters

If your filter accumulates data across items, add a `flush()` method:

```python
class StatsFilter(Filter["Mesh"]):
    def __init__(self, output: str) -> None:
        self._output = output
        self._rows: list[dict] = []

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        for mesh in items:
            self._rows.append(self._compute(mesh))
            yield mesh

    def flush(self) -> str | None:
        """Write accumulated data. Called after pipeline execution."""
        if not self._rows:
            return None
        self._write(self._rows, self._output)
        return self._output
```

The interactive CLI calls `flush()` automatically on any filter that has it.

## Custom Sink

A sink persists items and returns file paths.  Subclass
{class}`~physicsnemo_curator.core.base.Sink`:

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from physicsnemo_curator.core.base import Sink, Param

if TYPE_CHECKING:
    from collections.abc import Generator
    from physicsnemo.mesh import Mesh

class VTKSink(Sink["Mesh"]):
    name: ClassVar[str] = "VTK Writer"
    description: ClassVar[str] = "Save meshes as VTK files"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="output_dir", description="Output directory", type=str),
        ]

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir

    def __call__(self, items: Generator[Mesh], index: int) -> list[str]:
        paths: list[str] = []
        for seq, mesh in enumerate(items):
            path = f"{self._output_dir}/mesh_{index:04d}_{seq}.vtu"
            self._save_vtk(mesh, path)
            paths.append(path)
        return paths
```

## Registering Components

To make components discoverable by the CLI, register them in your
submodule's `__init__.py`:

```python
# src/physicsnemo_curator/mymodule/__init__.py
from physicsnemo_curator.core.registry import registry
from physicsnemo_curator.core.store import LocalFileStore, FsspecFileStore

from .sources.my_source import MySource
from .filters.my_filter import MyFilter
from .sinks.my_sink import MySink

registry.register_submodule(
    "mymodule",
    "My custom data processing",
    "some_dependency",  # import check for availability
)
registry.register_store("mymodule", "Local directory", LocalFileStore)
registry.register_store("mymodule", "Remote (fsspec)", FsspecFileStore)
registry.register_source("mymodule", MySource)
registry.register_filter("mymodule", MyFilter)
registry.register_sink("mymodule", MySink)
```

The CLI will discover the submodule when `physicsnemo_curator.mymodule` is imported.

## Testing

Use the `requires` marker to skip tests when optional dependencies are
missing:

```python
import pytest

pytestmark = pytest.mark.requires("mesh")

from physicsnemo_curator.core.store import LocalFileStore
from physicsnemo_curator.mesh.sources.vtk import VTKSource

class TestMySource:
    def test_len(self, tmp_path):
        # Create test fixtures...
        store = LocalFileStore(str(tmp_path))
        source = MySource(store=store)
        assert len(source) > 0

    def test_yields_correct_type(self, tmp_path):
        store = LocalFileStore(str(tmp_path))
        source = MySource(store=store)
        item = next(source[0])
        assert isinstance(item, Mesh)
```

Run tests with:

```bash
make test           # All tests
make test-core      # Core tests only (no optional deps)
make test-mesh      # Mesh tests only
make test-unit      # Unit tests
make test-integration  # Integration tests
make test-e2e       # End-to-end tests
```

## Executing Components

Once components are registered, use {func}`~physicsnemo_curator.run.run_pipeline`
to execute a pipeline efficiently:

```python
from physicsnemo_curator import run_pipeline

pipeline = (
    MySource(store=store)
    .filter(MyFilter())
    .write(MySink(output_dir="./out/"))
)

# Sequential
results = run_pipeline(pipeline)

# Parallel — uses all CPUs with the best available backend
results = run_pipeline(pipeline, n_jobs=-1)
```

See {doc}`/user-guide/parallel` for backend options and process-isolation
considerations.
