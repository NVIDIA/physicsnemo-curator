<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Extending PhysicsNeMo Curator

PhysicsNeMo Curator is designed around the **Pipes and Filters** pattern.
Every pipeline is assembled from three building blocks:

| Component | Role | Base class |
|-----------|------|------------|
| **Source** | Reads raw data and yields domain objects | {class}`~physicsnemo_curator.core.base.Source` |
| **Filter** | Transforms a stream of objects | {class}`~physicsnemo_curator.core.base.Filter` |
| **Sink** | Persists objects and returns the paths it wrote | {class}`~physicsnemo_curator.core.base.Sink` |

You can create custom versions of any of these to support your own datasets,
transformations, or output formats — no changes to the core library required.

## Anatomy of a Component

Every component shares the same skeleton:

1. **`name`** and **`description`** class variables — used by the CLI and
   registry for display.
2. A **`params()`** classmethod returning a list of
   {class}`~physicsnemo_curator.core.base.Param` — drives CLI prompts and
   documents the constructor interface.
3. An **`__init__`** accepting those parameters as keyword arguments.
4. A core method implementing the component's logic (`__getitem__`,
   `__call__`, or `__call__` depending on the component type).

```python
from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING
from physicsnemo_curator.core.base import Param

if TYPE_CHECKING:
    pass  # domain-specific imports here

class MyComponent:
    name: ClassVar[str] = "My Component"
    description: ClassVar[str] = "One-line description of what it does"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="option", description="Processing option", type=str, default="default"),
        ]

    def __init__(self, option: str = "default") -> None:
        self._option = option
```

## FileStores

A {class}`~physicsnemo_curator.core.store.FileStore` maps integer indices to
local file paths.  Sources never handle file discovery or downloads directly —
that is the store's job.  Any object with `__len__` and `__getitem__`
(returning a `str` path) satisfies the protocol:

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

Register custom stores with the global registry so the interactive CLI can
offer them as a data-source option:

```python
from physicsnemo_curator.core.registry import registry

registry.register_store("mesh", "My Database", DatabaseFileStore)
```

The CLI will then show *My Database* alongside *Local directory* and
*Remote (fsspec)* in the store selection prompt.

Built-in stores include {class}`~physicsnemo_curator.core.store.LocalFileStore`,
{class}`~physicsnemo_curator.core.store.FsspecFileStore`, and
{class}`~physicsnemo_curator.core.store.RunIndexedFileStore`.

## Registering Components

To make components discoverable by the CLI, register them in your submodule's
`__init__.py`:

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

## Executing Pipelines

Once components are registered, use {func}`~physicsnemo_curator.run.run_pipeline`
to execute a pipeline:

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

## What's Next

```{toctree}
:maxdepth: 1

sources
filters
sinks
```
