---
name: add-filter
description: >
  Guide for adding a new filter to PhysicsNeMo Curator. Covers discovery
  questions, implementation patterns (pass-through, stateful, in-place),
  generator semantics, testing, and registration.
---

# Adding a New Filter

This skill walks through adding a new `Filter` to PhysicsNeMo Curator,
from initial design through implementation, testing, and registration.

## Step 0: Gather Requirements

Before writing any code, answer these questions. Ask the user if anything
is unclear.

### Domain

Which submodule does this filter belong to?

| Domain | Type parameter | Submodule | Dependency group |
|--------|---------------|-----------|-----------------|
| **mesh** | `Filter["Mesh"]` | `src/physicsnemo_curator/mesh/` | `mesh` (physicsnemo, pyvista, pyarrow, torch) |
| **da** | `Filter["xr.DataArray"]` | `src/physicsnemo_curator/da/` | `da` (xarray, earth2studio, zarr) |

### Filter Design

1. **Purpose** -- What does this filter do? (compute statistics, convert
   precision, extract metadata, transform data, validate, etc.)
2. **Pass-through or transforming?** -- Does it yield the input unchanged
   (side-effect only), or does it modify the data before yielding?
3. **Stateful?** -- Does the filter accumulate state across items?
   (e.g. running statistics, row accumulation for output files)
4. **Output** -- Does it produce a side-effect output? If so, what format?
   (Parquet, JSON-lines, Zarr, logging only, none)
5. **Flush required?** -- If stateful, does it need a `flush()` method to
   write accumulated results at the end?
6. **Parameters** -- What user-configurable parameters does the filter
   need? (output path, dtype, thresholds, flags, etc.)
7. **Cardinality** -- Does it yield exactly 1 item per input, or can it
   expand (1-to-many) or contract (many-to-fewer)?
8. **Helper logic** -- Does it need module-level helper functions or
   private helper classes? (e.g. recursive TensorDict traversal,
   accumulator objects)
9. **Dependencies** -- Does it require additional imports beyond the domain
   group? (e.g. specific PyArrow, h5py, scipy modules)
10. **Pipeline position** -- Where in a typical pipeline would this filter
    go? (early for validation, middle for transforms, late for statistics)

### Reference: Existing Filters

| Filter | File | Type | Pattern | Good example for |
|--------|------|------|---------|-----------------|
| `MeshInfoFilter` | `mesh/filters/mesh_info.py` | Mesh | Pass-through + logging + JSON-lines | Optional output, log levels |
| `MeanFilter` | `mesh/filters/mean.py` | Mesh | Pass-through + accumulate rows | Parquet output, simple flush |
| `PrecisionFilter` | `mesh/filters/precision.py` | Mesh | In-place modification | Dtype conversion, validation in init |
| `StatsFilter` | `mesh/filters/stats.py` | Mesh | Pass-through + Welford accumulator | Complex statistics, merge function |
| `MomentsFilter` | `da/filters/moments.py` | DataArray | Pass-through + running moments | Zarr output, helper class |

## Step 1: Create the Filter Module

Create the filter file at the appropriate location:

- **mesh**: `src/physicsnemo_curator/mesh/filters/<name>.py`
- **da**: `src/physicsnemo_curator/da/filters/<name>.py`

### Required SPDX Header

Every file MUST start with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### Import Block Template

```python
"""Brief description of what this filter does."""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, ClassVar

# Third-party imports as needed (torch, pyarrow, numpy, etc.)

from physicsnemo_curator.core.base import Filter, Param

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh  # or: import xarray as xr

logger = logging.getLogger(__name__)
```

### Filter Class Template

```python
class <ClassName>(Filter["Mesh"]):
    """Detailed description of the filter.

    <2-3 sentence description of what this filter computes or transforms,
    when you would use it in a pipeline, and what output it produces (if any).>

    Parameters
    ----------
    param1 : str
        Description of first parameter.
    param2 : int
        Description of second parameter.

    Examples
    --------
    >>> filt = <ClassName>(param1="value")  # doctest: +SKIP
    >>> pipeline = source.filter(filt).write(sink)  # doctest: +SKIP
    """

    name: ClassVar[str] = "<Display Name>"
    description: ClassVar[str] = "<Short one-line description for CLI>"

    @classmethod
    def params(cls) -> list[Param]:
        """Return configurable parameters for this filter.

        Returns
        -------
        list[Param]
            Parameter descriptors.
        """
        return [
            Param(
                name="param1",
                description="Description of param1",
                type=str,
                default="default_value",
            ),
        ]

    def __init__(self, param1: str = "default_value") -> None:
        """Initialize the filter.

        Parameters
        ----------
        param1 : str
            First parameter.
        """
        self._param1 = param1
        # Validate parameters in __init__

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        """Process a stream of meshes.

        Parameters
        ----------
        items : Generator[Mesh]
            Stream of incoming meshes.

        Yields
        ------
        Mesh
            Processed meshes.
        """
        for mesh in items:
            # Process mesh here
            yield mesh
```

### Generator Semantics

The `__call__` method receives and returns a `Generator[T]`. The standard
pattern is:

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        # ... process ...
        yield mesh
```

Key rules:

- **Always yield** -- filters must yield items (even if unmodified) for
  downstream filters and sinks to receive them
- **Lazy processing** -- iterate the input generator lazily, do not
  consume the whole stream upfront unless absolutely necessary
- **1-to-1 is most common** -- every existing filter yields exactly one
  item per input item
- **1-to-many is allowed** -- yield multiple items per input if needed
  (e.g. splitting a multi-block mesh into individual blocks)
- **Filtering (0-to-1) is allowed** -- skip yielding for items that
  fail a condition

### Implementation Patterns

#### Pattern A: Pass-Through (Side-Effect Only)

The filter examines but does not modify the data. Used for logging,
metadata extraction, and statistics accumulation.

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        info = self._extract_info(mesh)
        self._log_info(info)
        self._rows.append(info)
        yield mesh  # unchanged
```

Examples: `MeshInfoFilter`, `MeanFilter`, `StatsFilter`, `MomentsFilter`

#### Pattern B: In-Place Modification

The filter modifies the data before yielding. Used for dtype conversion,
field renaming, coordinate transforms.

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        if mesh.point_data is not None:
            self._convert_fields(mesh.point_data)
        yield mesh  # modified in-place
```

Example: `PrecisionFilter`

#### Pattern C: Stateful with Flush

The filter accumulates results across all items and writes output at the
end. Requires a `flush()` method.

```python
def __init__(self, output: str) -> None:
    self._output_path = pathlib.Path(output)
    self._rows: list[dict[str, object]] = []

def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        row = self._compute_row(mesh)
        self._rows.append(row)
        yield mesh

def flush(self) -> str | None:
    """Write accumulated results to output file.

    Returns
    -------
    str or None
        Path of written file, or ``None`` if nothing to write.
    """
    if not self._rows:
        return None
    # Write self._rows to self._output_path
    return str(self._output_path)
```

Examples: `MeanFilter` (Parquet), `StatsFilter` (Parquet),
`MeshInfoFilter` (JSON-lines), `MomentsFilter` (Zarr)

### Output Format Patterns

#### Parquet Output (MeanFilter / StatsFilter pattern)

```python
import pyarrow as pa
import pyarrow.parquet as pq

def flush(self) -> str | None:
    if not self._rows:
        return None
    table = pa.Table.from_pylist(self._rows)
    pq.write_table(table, str(self._output_path))
    return str(self._output_path)
```

#### JSON-Lines Output (MeshInfoFilter pattern)

```python
import json

def __init__(self, output: str | None = None) -> None:
    self._file_handle: TextIO | None = None
    if output:
        self._output_path = pathlib.Path(output)

def _write_to_file(self, info: dict) -> None:
    if self._file_handle is None and hasattr(self, "_output_path"):
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = self._output_path.open("w")
    if self._file_handle is not None:
        self._file_handle.write(json.dumps(info) + "\n")

def flush(self) -> str | None:
    if self._file_handle is not None:
        self._file_handle.close()
        return str(self._output_path)
    return None
```

#### Zarr Output (MomentsFilter pattern)

```python
import zarr

def flush(self) -> str | None:
    ds = self._finalize()
    ds.to_zarr(str(self._output_path), mode="w")
    return str(self._output_path)
```

### Helper Function Patterns

Module-level private helpers for reusable logic:

```python
def _extract_leaf_tensors(
    td: object, prefix: str = ""
) -> list[tuple[str, torch.Tensor]]:
    """Recursively extract leaf tensors from a TensorDict.

    Parameters
    ----------
    td : object
        TensorDict-like object.
    prefix : str
        Key prefix for nested access.

    Returns
    -------
    list[tuple[str, torch.Tensor]]
        Pairs of (key_path, tensor).
    """
    results = []
    for key in td.keys():
        child = td[key]
        full_key = f"{prefix}{key}"
        if hasattr(child, "keys"):
            results.extend(_extract_leaf_tensors(child, f"{full_key}/"))
        else:
            results.append((full_key, child))
    return results
```

Helper classes for complex state:

```python
class _MyAccumulator:
    """Internal accumulator for running computations."""

    def __init__(self) -> None:
        self._count = 0
        # ... state ...

    def update(self, value: torch.Tensor) -> None:
        """Incorporate a new observation."""
        self._count += 1
        # ... update state ...

    def finalize(self) -> dict[str, float]:
        """Return final computed values."""
        return {"count": self._count, ...}
```

### Param Declaration Patterns

**Required parameter** (no default -- user must provide):

```python
Param(name="output", description="Output file path", type=str)
```

**Optional with default**:

```python
Param(name="log_level", description="Logging level", type=str,
      default="info", choices=["info", "debug"])
```

**Boolean flag**:

```python
Param(name="include_fields", description="Include field details",
      type=bool, default=True)
```

**With constrained choices**:

```python
Param(name="target_dtype", description="Target floating-point dtype",
      type=str, default="float32",
      choices=["float32", "float16", "bfloat16"])
```

**None-able** (optional output):

```python
Param(name="output", description="Optional output path", type=str,
      default=None)
```

## Step 2: Write Tests

Create the test file at `test/<domain>/test_<name>.py`.

### Test File Structure

```python
"""Tests for <ClassName>."""

# SPDX header (same as source files)

from __future__ import annotations

import pathlib

import numpy as np
import pyvista as pv
import pytest

pytestmark = pytest.mark.requires("<domain>")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _create_test_mesh(
    directory: pathlib.Path, name: str = "test.vtu"
) -> pathlib.Path:
    """Create a simple VTK unstructured grid for testing.

    Returns the path to the written file.
    """
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
    )
    cells = np.array([[3, 0, 1, 2], [3, 0, 2, 3]])
    cell_types = np.array([5, 5])  # VTK_TRIANGLE

    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data["temperature"] = np.array([100.0, 200.0, 300.0, 400.0])
    grid.point_data["pressure"] = np.array([1.0, 2.0, 3.0, 4.0])
    grid.cell_data["velocity"] = np.array([10.0, 20.0])

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    grid.save(str(path))
    return path
```

Alternatively, for filters that do not need VTK files, you can construct
`Mesh` objects directly using mock data (see the `NavierStokesCylinderSource`
test pattern in `test/mesh/test_ns_cylinder.py`).

### Test Classes

#### Unit Tests (Metadata)

```python
class Test<ClassName>Unit:
    """Metadata tests."""

    def test_params_list(self) -> None:
        from <module> import <ClassName>

        params = <ClassName>.params()
        assert len(params) >= 0
        names = [p.name for p in params]
        # Assert expected parameter names

    def test_name_and_description(self) -> None:
        from <module> import <ClassName>

        assert isinstance(<ClassName>.name, str)
        assert len(<ClassName>.name) > 0
        assert len(<ClassName>.description) > 0
```

#### Pass-Through Tests

```python
@pytest.mark.integration
class Test<ClassName>Integration:
    """Tests against local test data."""

    def test_yields_mesh_unchanged(self, tmp_path: pathlib.Path) -> None:
        """Filter should yield the mesh without modification."""
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        _create_test_mesh(tmp_path / "vtk")
        source = VTKSource.from_path(str(tmp_path / "vtk"))
        mesh_before = next(source[0])

        filt = <ClassName>(...)

        def gen():
            yield mesh_before

        meshes_out = list(filt(gen()))
        assert len(meshes_out) == 1
        assert meshes_out[0] is mesh_before  # identity check
```

#### Logging Tests

```python
    def test_logs_output(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Filter logs expected information."""
        import logging

        _create_test_mesh(tmp_path / "vtk")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        filt = <ClassName>(...)

        with caplog.at_level(logging.INFO):
            list(filt(source[0]))

        assert "expected text" in caplog.text
```

#### Output File Tests

```python
    def test_writes_output_file(self, tmp_path: pathlib.Path) -> None:
        """Filter writes expected output."""
        _create_test_mesh(tmp_path / "vtk")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        output_path = tmp_path / "output.parquet"
        filt = <ClassName>(output=str(output_path))
        list(filt(source[0]))
        filt.flush()

        assert output_path.exists()
        # Verify output content
```

#### Computation Verification Tests

```python
    def test_computes_correct_values(self, tmp_path: pathlib.Path) -> None:
        """Filter produces correct computed results."""
        _create_test_mesh(tmp_path / "vtk")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        output_path = tmp_path / "stats.parquet"
        filt = <ClassName>(output=str(output_path))
        list(filt(source[0]))
        filt.flush()

        # Read output and verify values
        import pyarrow.parquet as pq

        table = pq.read_table(str(output_path))
        # Assert on specific computed values with tolerance
        # e.g. assert abs(value - 250.0) < 1e-5
```

#### Pipeline Integration Tests

```python
@pytest.mark.e2e
class Test<ClassName>Pipeline:
    """End-to-end pipeline tests."""

    def test_chained_in_pipeline(self, tmp_path: pathlib.Path) -> None:
        """Filter works correctly in a multi-filter pipeline."""
        from physicsnemo_curator.mesh.sinks.mesh_writer import MeshSink
        from physicsnemo_curator.mesh.sources.vtk import VTKSource

        vtk_dir = tmp_path / "vtk"
        _create_test_mesh(vtk_dir, "mesh_0.vtu")
        _create_test_mesh(vtk_dir, "mesh_1.vtu")

        output_dir = tmp_path / "output"

        filt = <ClassName>(...)

        pipeline = (
            VTKSource.from_path(str(vtk_dir))
            .filter(filt)
            .write(MeshSink(output_dir=str(output_dir)))
        )

        assert len(pipeline) == 2

        for i in range(len(pipeline)):
            paths = pipeline[i]
            assert len(paths) == 1
            assert pathlib.Path(paths[0]).exists()

        # Flush if stateful
        # filt.flush()
```

#### Registry Tests

```python
class Test<ClassName>Registry:
    """Test that the filter is registered."""

    def test_filter_registered(self) -> None:
        from physicsnemo_curator.core.registry import registry

        names = [f.name for f in registry.list_filters("<domain>")]
        assert "<Display Name>" in names
```

### Test Markers Reference

| Marker | Purpose |
|--------|---------|
| `pytestmark = pytest.mark.requires("mesh")` | Module-level: skip all if mesh deps missing |
| `@pytest.mark.requires("da")` | Skip if da dependencies not installed |
| `@pytest.mark.integration` | Tests that touch filesystem |
| `@pytest.mark.e2e` | End-to-end pipeline tests |
| `@pytest.mark.slow` | Slow tests, excluded from quick CI |

### Running Tests

```bash
# Unit and integration tests (fast, no network)
uv run pytest test/<domain>/test_<name>.py -v

# Pipeline integration test
uv run pytest test/<domain>/test_<name>.py -v -k "Pipeline"

# Full test suite to check for regressions
uv run pytest test/<domain>/ -v -k "not slow"
```

## Step 3: Register the Filter

### Edit the domain `__init__.py`

For **mesh** filters, edit `src/physicsnemo_curator/mesh/__init__.py`:

```python
# Add import (alphabetical order among filters)
from physicsnemo_curator.mesh.filters.<module> import <ClassName>

# Add registration (after existing register_filter calls)
registry.register_filter("mesh", <ClassName>)

# Add to __all__ (alphabetical order)
__all__ = [
    ...,
    "<ClassName>",
    ...,
]
```

For **da** filters, edit `src/physicsnemo_curator/da/__init__.py` with
the same pattern using `"da"` as the submodule name.

### Export any public helpers

If the filter provides a public helper function (like `merge_welford_stats`
in `StatsFilter`), add it to the import and `__all__` as well:

```python
from physicsnemo_curator.mesh.filters.<module> import <ClassName>, <helper_func>

__all__ = [
    ...,
    "<ClassName>",
    "<helper_func>",
    ...,
]
```

## Step 4: Quality Checks

Run all checks before committing:

```bash
# Format
uv run ruff format \
  src/physicsnemo_curator/<domain>/filters/<name>.py \
  test/<domain>/test_<name>.py

# Lint
uv run ruff check --fix \
  src/physicsnemo_curator/<domain>/filters/<name>.py \
  test/<domain>/test_<name>.py

# Docstring coverage (must be >= 99%)
uv run interrogate

# Type checking
uv run ty check

# Run filter tests
uv run pytest test/<domain>/test_<name>.py -v

# Run full domain test suite for regressions
uv run pytest test/<domain>/ -v -k "not slow"
```

## Step 5: Commit

Use the `commit` tool (or follow Conventional Commits format):

```text
feat(<domain>): add <ClassName> for <purpose>

<Optional body describing what the filter computes and what output it produces.>
```

## Checklist

Before considering the filter complete, verify:

- [ ] SPDX license headers on all new files
- [ ] Filter inherits from `Filter["Mesh"]` or `Filter["xr.DataArray"]`
- [ ] `name` and `description` ClassVars are set
- [ ] `params()` returns all configurable parameters
- [ ] `__init__()` validates parameter constraints
- [ ] `__call__()` receives `Generator[T]`, yields `Generator[T]`
- [ ] Generator is consumed lazily (no upfront `list()` conversion)
- [ ] Pass-through: yields items for downstream consumption
- [ ] NumPy-style docstrings on class, `__init__`, `__call__`, `flush` (if any)
- [ ] `flush()` method implemented if filter accumulates state
- [ ] Module-level `logger = logging.getLogger(__name__)` if logging
- [ ] Helper functions/classes have docstrings and type annotations
- [ ] `from __future__ import annotations` at top
- [ ] `TYPE_CHECKING` block for `Generator` and domain type imports
- [ ] Registered in domain `__init__.py` with `registry.register_filter()`
- [ ] Added to `__all__` in domain `__init__.py`
- [ ] Public helpers also exported in `__all__`
- [ ] Unit tests: params, name, description
- [ ] Integration tests: pass-through, output files, computed values
- [ ] Pipeline test: filter works in a chained pipeline (marked `@pytest.mark.e2e`)
- [ ] Registry test: filter is discoverable
- [ ] `ruff format` clean
- [ ] `ruff check` clean
- [ ] `interrogate` >= 99%
- [ ] `ty check` clean
- [ ] All tests pass
- [ ] No regressions in existing tests

## Reference: Filter[T] ABC

From `src/physicsnemo_curator/core/base.py`:

```python
class Filter[T](ABC):
    """Abstract filter/transform that processes a stream of T items."""

    name: ClassVar[str]
    description: ClassVar[str]

    @classmethod
    @abstractmethod
    def params(cls) -> list[Param]: ...

    @abstractmethod
    def __call__(self, items: Generator[T]) -> Generator[T]: ...
```

Key differences from `Source[T]`:

- Sources have `__len__` and `__getitem__` -- filters do not
- Filters receive a generator and return a generator (`__call__`)
- Sources are indexed by integer -- filters process streams
