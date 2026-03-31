# Writing Custom Components

This guide explains how to create custom sources, filters, and sinks for
PhysicsNeMo Curator.

## Anatomy of a Component

Every component has:

1. **`name`** and **`description`** class variables for CLI display
2. A **`params()`** classmethod returning a list of {class}`~curator.core.base.Param`
3. An **`__init__`** accepting those parameters as keyword arguments
4. A core method (`__getitem__`, `__call__`, or `__call__`) implementing the logic

## Custom Source

A source reads data and yields items.  Subclass
{class}`~curator.core.base.Source` and implement `__len__`, `__getitem__`,
and `params`:

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from curator.core.base import Source, Param

if TYPE_CHECKING:
    from collections.abc import Generator
    from physicsnemo.mesh import Mesh

class MySource(Source["Mesh"]):
    name: ClassVar[str] = "My Reader"
    description: ClassVar[str] = "Reads data from my custom format"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="input_path", description="Path to data", type=str),
        ]

    def __init__(self, input_path: str) -> None:
        self._path = input_path
        self._items = self._discover()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Generator[Mesh]:
        item = self._items[index]
        mesh = self._load(item)
        yield mesh

    def _discover(self) -> list[str]:
        """Find items to process."""
        ...

    def _load(self, item: str) -> Mesh:
        """Load a single item."""
        ...
```

Key points:

- `__getitem__` is a **generator** (uses `yield`).  It can yield multiple
  items per index if needed.
- `params()` drives the CLI prompts and documents the constructor interface.

## Custom Filter

A filter transforms a stream of items.  Subclass
{class}`~curator.core.base.Filter`:

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from curator.core.base import Filter, Param

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
{class}`~curator.core.base.Sink`:

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from curator.core.base import Sink, Param

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
# src/curator/mymodule/__init__.py
from curator.core.registry import registry

from .sources.my_source import MySource
from .filters.my_filter import MyFilter
from .sinks.my_sink import MySink

registry.register_submodule(
    "mymodule",
    "My custom data processing",
    "some_dependency",  # import check for availability
)
registry.register_source("mymodule", MySource)
registry.register_filter("mymodule", MyFilter)
registry.register_sink("mymodule", MySink)
```

The CLI will discover the submodule when `curator.mymodule` is imported.

## Testing

Use pytest with `pytest.importorskip` for optional dependencies:

```python
import pytest

pv = pytest.importorskip("pyvista")
torch = pytest.importorskip("torch")

from curator.mesh.sources.vtk import VTKSource

class TestMySource:
    def test_len(self, tmp_path):
        # Create test fixtures...
        source = MySource(input_path=str(tmp_path))
        assert len(source) > 0

    def test_yields_correct_type(self, tmp_path):
        source = MySource(input_path=str(tmp_path))
        item = next(source[0])
        assert isinstance(item, Mesh)
```

Run tests with:

```bash
make test
# or
uv run pytest test/ -v
```
