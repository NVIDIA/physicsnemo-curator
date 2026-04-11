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

## Registering Components

To make components discoverable by the CLI, register them in your submodule's
`__init__.py`:

```python
# src/physicsnemo_curator/mymodule/__init__.py
from physicsnemo_curator.core.registry import registry

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

## Testing

Use the `requires` marker to skip tests when optional dependencies are
missing:

```python
import pytest

pytestmark = pytest.mark.requires("mesh")

from physicsnemo_curator.mesh.sources.vtk import VTKSource

class TestMySource:
    def test_len(self, tmp_path):
        # Create test fixtures in tmp_path...
        source = MySource(input_path=str(tmp_path))
        assert len(source) > 0

    def test_yields_correct_type(self, tmp_path):
        # Create test fixtures in tmp_path...
        source = MySource(input_path=str(tmp_path))
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
    MySource(input_path="./data/")
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
