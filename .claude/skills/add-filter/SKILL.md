---
name: add-filter
description: >
  Guide for adding a new filter to PhysicsNeMo Curator. Covers discovery
  questions, implementation patterns (pass-through, stateful, in-place),
  generator semantics, flush/artifacts, dashboard widgets, merge for
  parallel workers, testing, and registration.
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
| **mesh** | `Filter["Mesh"]` | `src/physicsnemo_curator/domains/mesh/` | `mesh` (physicsnemo, pyvista, pyarrow, torch) |
| **da** | `Filter["xr.DataArray"]` | `src/physicsnemo_curator/domains/da/` | `da` (xarray, earth2studio, zarr) |
| **atm** | `Filter["AtomicData"]` | `src/physicsnemo_curator/domains/atm/` | `atm` (nvalchemi, ase, torch) |

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
6. **Artifacts?** -- If it writes files, should it implement `artifacts()`
   to report paths to the pipeline store?
7. **Dashboard widget?** -- Should it provide a `dashboard_panel()` for
   interactive visualization of its output?
8. **Parallel merge?** -- If stateful, does it need a `merge()` static
   method to combine per-worker output files?
9. **Parameters** -- What user-configurable parameters does the filter
   need? (output path, dtype, thresholds, flags, etc.)
10. **Cardinality** -- Does it yield exactly 1 item per input, or can it
    expand (1-to-many) or contract (many-to-fewer)?
11. **Helper logic** -- Does it need module-level helper functions or
    private helper classes? (e.g. recursive TensorDict traversal,
    accumulator objects)
12. **Dependencies** -- Does it require additional imports beyond the domain
    group? (e.g. specific PyArrow, h5py, scipy modules)
13. **Pipeline position** -- Where in a typical pipeline would this filter
    go? (early for validation, middle for transforms, late for statistics)

### Reference: Existing Filters

| Filter | File | Domain | Pattern | Good example for |
|--------|------|--------|---------|-----------------|
| `MeshInfoFilter` | `mesh/filters/mesh_info.py` | Mesh | Pass-through + logging + JSON-lines | Optional output, log levels |
| `MeanFilter` | `mesh/filters/mean.py` | Mesh | Pass-through + accumulate rows | Parquet output, simple flush |
| `PrecisionFilter` | `mesh/filters/precision.py` | Mesh | In-place modification | Dtype conversion, validation |
| `FieldSelectFilter` | `mesh/filters/field_select.py` | Mesh | In-place modification | Include/exclude filtering |
| `RandomPermutationFilter` | `mesh/filters/random_permutation.py` | Mesh | In-place modification | Stateless data shuffling |
| `EdgeComputeFilter` | `mesh/filters/edge_compute.py` | Mesh | In-place modification | Adding computed data to mesh |
| `WallNodeFilter` | `mesh/filters/wall_node.py` | Mesh | In-place modification | Complex geometric filtering |
| `MeshQualityFilter` | `mesh/filters/quality.py` | Mesh | Stateful + flush + merge | Quality report, Parquet output |
| `MeshStatsFilter` | `mesh/filters/stats.py` | Mesh | Stateful + flush + merge + dashboard | Full pattern, Welford stats |
| `DataArrayStatsFilter` | `da/filters/stats.py` | DataArray | Stateful + flush + merge + dashboard | Zarr output, online moments |
| `AtomicInfoFilter` | `atm/filters/atomic_info.py` | AtomicData | Pass-through + logging | Metadata extraction |
| `AtomicStatsFilter` | `atm/filters/stats.py` | AtomicData | Stateful + flush + merge + dashboard | Atomic-level statistics |

## Step 1: Create the Filter Module

Create the filter file at the appropriate location:

- **mesh**: `src/physicsnemo_curator/domains/mesh/filters/<name>.py`
- **da**: `src/physicsnemo_curator/domains/da/filters/<name>.py`
- **atm**: `src/physicsnemo_curator/domains/atm/filters/<name>.py`

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

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

# Third-party imports as needed (torch, pyarrow, numpy, etc.)

from physicsnemo_curator.core.base import Filter, Param
from physicsnemo_curator.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicsnemo.mesh import Mesh  # or: import xarray as xr
```

**Note**: Prefer `get_logger(self)` from `physicsnemo_curator.core.logging`
over raw `logging.getLogger(__name__)`. It provides worker-aware structured
logging with timing and is captured by the TUI/database.

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
        self._log = get_logger(self)
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
        self._log.info("Processed mesh: %s", info["name"])
        self._rows.append(info)
        yield mesh  # unchanged
```

Examples: `MeshInfoFilter`, `MeanFilter`, `MeshStatsFilter`,
`DataArrayStatsFilter`, `AtomicInfoFilter`

#### Pattern B: In-Place Modification

The filter modifies the data before yielding. Used for dtype conversion,
field selection, coordinate transforms, edge computation.

```python
def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
    for mesh in items:
        if mesh.point_data is not None:
            self._convert_fields(mesh.point_data)
        yield mesh  # modified in-place
```

Examples: `PrecisionFilter`, `FieldSelectFilter`, `RandomPermutationFilter`,
`EdgeComputeFilter`, `WallNodeFilter`

#### Pattern C: Stateful with Flush

The filter accumulates results across items and writes output when
flushed. Requires `flush()`, `artifacts()`, and `_output_path`.

```python
def __init__(self, output: str) -> None:
    self._log = get_logger(self)
    self._output_path = pathlib.Path(output)
    self._rows: list[dict[str, object]] = []
    self._last_artifacts: list[str] = []

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
    self._output_path.parent.mkdir(parents=True, exist_ok=True)
    # ... write data ...
    path = str(self._output_path)
    self._rows.clear()
    self._last_artifacts = [path]
    return path

def artifacts(self) -> list[str]:
    """Return paths written by the last flush call.

    Returns
    -------
    list[str]
        Paths of files written since the last call, or ``[]``.
    """
    paths = self._last_artifacts
    self._last_artifacts = []
    return paths
```

Examples: `MeshQualityFilter` (Parquet), `MeanFilter` (Parquet),
`MeshInfoFilter` (JSON-lines)

#### Pattern D: Stateful + Flush + Merge + Dashboard (Full Pattern)

The complete pattern for parallel-aware statistics filters with
interactive visualization. Adds `merge()` and `dashboard_panel()`.

```python
class MyStatsFilter(Filter["Mesh"]):
    name: ClassVar[str] = "My Statistics"
    description: ClassVar[str] = "Compute statistics and save to Parquet"

    def __init__(self, output: str) -> None:
        self._log = get_logger(self)
        self._output_path = pathlib.Path(output)
        self._rows: list[dict[str, object]] = []
        self._last_artifacts: list[str] = []

    def __call__(self, items: Generator[Mesh]) -> Generator[Mesh]:
        for mesh in items:
            self._rows.extend(self._compute_stats(mesh))
            yield mesh

    def flush(self) -> str | None:
        """Write accumulated stats; append if file exists."""
        if not self._rows:
            return None
        # ... write/append to Parquet ...
        path = str(self._output_path)
        self._rows.clear()
        self._last_artifacts = [path]
        return path

    def artifacts(self) -> list[str]:
        """Return paths written by last flush."""
        paths = self._last_artifacts
        self._last_artifacts = []
        return paths

    @staticmethod
    def merge(parquet_paths: list[str], output: str) -> str:
        """Concatenate per-worker Parquet files into one.

        Parameters
        ----------
        parquet_paths : list[str]
            Paths to per-worker output files.
        output : str
            Path for the merged output.

        Returns
        -------
        str
            Path of written merged file.
        """
        # ... concatenate tables ...
        return output

    @classmethod
    def dashboard_panel(
        cls,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> Any:
        """Return interactive Panel widget for visualizing artifacts.

        Parameters
        ----------
        artifact_paths : list[str]
            Paths to artifact files produced by this filter.
        selected_index : int or None
            Currently selected pipeline index, if any.

        Returns
        -------
        pn.viewable.Viewable or None
            A Panel component, or None if no widget.
        """
        import holoviews as hv
        import panel as pn
        # ... build interactive visualization ...
        return pn.Row(sidebar, plot_pane, sizing_mode="stretch_width")

    @classmethod
    def dashboard_layout_hints(cls) -> dict[str, int]:
        """Declare grid space for dashboard GridStack placement.

        Returns
        -------
        dict[str, int]
            cols: GridStack columns (1-12), rows: rows (1+).
        """
        return {"cols": 12, "rows": 3}
```

Examples: `MeshStatsFilter`, `DataArrayStatsFilter`, `AtomicStatsFilter`

### Pipeline Runner Integration (flush / artifacts)

The pipeline runner (`run_pipeline`) automatically manages stateful
filters. Understanding this is critical for correct implementation:

1. **Auto-flush**: After each pipeline index is processed, the runner
   calls `flush()` on every filter that has both a `flush` method and
   an `_output_path` attribute.

2. **Worker-specific paths**: The runner rewrites `_output_path` to
   include a worker ID suffix for process-based parallel backends:
   - If the path contains `{worker_id}`, it's used as a template
   - Otherwise, the path is rewritten as `{stem}_worker_{pid}{suffix}`

3. **Artifact tracking**: After flushing, if the pipeline has
   `track_metrics=True`, the runner calls `artifacts()` and records
   the paths in the pipeline database.

4. **Append semantics**: Because `flush()` is called after **each
   index**, filters should **append** to existing files rather than
   overwrite. This is important — each flush adds the current index's
   data to the worker's output file.

**Convention**: A filter's `_output_path` attribute is the contract with
the runner. If your filter has `flush()` but no `_output_path`, the
runner will skip it.

```python
# The runner does essentially this after each index:
for f in pipeline.filters:
    if hasattr(f, "flush") and hasattr(f, "_output_path"):
        # Rewrite path for this worker
        f._output_path = worker_specific_path
        f.flush()
        # Record artifacts
        artifact_paths = f.artifacts()
```

### Flush Design Principles

1. **Return value**: `flush()` returns `str | None` — the path written,
   or `None` if nothing to write.

2. **Append on subsequent calls**: Since the runner calls `flush()` after
   each index, your flush should append new data to the existing file:

   ```python
   def flush(self) -> str | None:
       if not self._rows:
           return None

       new_table = pa.table(...)
       self._output_path.parent.mkdir(parents=True, exist_ok=True)

       # Append to existing file if present
       if self._output_path.exists():
           existing = pq.read_table(str(self._output_path))
           combined = pa.concat_tables([existing, new_table])
           pq.write_table(combined, str(self._output_path))
       else:
           pq.write_table(new_table, str(self._output_path))

       path = str(self._output_path)
       self._rows.clear()
       self._last_artifacts = [path]
       return path
   ```

3. **Clear state after flush**: Always clear the in-memory accumulator
   (`self._rows.clear()`) after writing.

4. **Update `_last_artifacts`**: Set the list of written paths so
   `artifacts()` can report them.

### The merge() Static Method

When running with parallel backends (`n_jobs > 1`), each worker writes
its own output file. After all workers finish, `run_pipeline` calls
`merge()` to combine them:

```python
@staticmethod
def merge(parquet_paths: list[str], output: str) -> str:
    """Concatenate per-worker output files.

    Parameters
    ----------
    parquet_paths : list[str]
        Paths to per-worker files.
    output : str
        Path for the merged file.

    Returns
    -------
    str
        Path of written merged file.

    Raises
    ------
    ValueError
        If parquet_paths is empty.
    """
    if not parquet_paths:
        msg = "parquet_paths must be a non-empty list."
        raise ValueError(msg)

    tables = [pq.read_table(p) for p in parquet_paths]
    merged = pa.concat_tables(tables)

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, str(out_path))
    return str(out_path)
```

The runner detects merge capability by checking for all three attributes:
`flush`, `_output_path`, and `merge`.

### Dashboard Widget Pattern

Filters can provide an interactive visualization widget for the
dashboard UI. This is a **classmethod** — it receives artifact paths
and returns a Panel component.

**When to implement**: If your filter writes output files (Parquet, Zarr)
that users would benefit from exploring visually.

**Key design principles**:

1. **Import Panel/HoloViews inside the method** — never at module level.
   Dashboard dependencies are optional.
2. **Return `pn.Row(sidebar, plot_pane)`** — standard layout with controls
   on the left and the visualization on the right.
3. **Handle missing data gracefully** — return `pn.pane.Markdown("*No data*")`
   when artifacts don't exist.
4. **Use `@pn.depends`** — for reactive widgets that update the plot.

```python
@classmethod
def dashboard_panel(
    cls,
    artifact_paths: list[str],
    selected_index: int | None = None,
) -> Any:
    """Return an interactive widget for this filter's artifacts.

    Parameters
    ----------
    artifact_paths : list[str]
        Paths to artifact files produced by this filter.
    selected_index : int or None
        Currently selected pipeline index, if any.

    Returns
    -------
    pn.viewable.Viewable or None
        A Panel component, or None if no widget.
    """
    import holoviews as hv
    import pandas as pd
    import panel as pn
    from bokeh.models import HoverTool

    hv.extension("bokeh")

    if not artifact_paths:
        return pn.pane.Markdown("*No artifacts found.*")

    # Load data from artifact files
    frames = []
    for path in artifact_paths:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:  # noqa: BLE001
            continue

    if not frames:
        return pn.pane.Markdown("*Could not read any artifacts.*")

    df = pd.concat(frames, ignore_index=True)

    # Create widgets
    x_select = pn.widgets.Select(name="X-Axis", options=columns, value="mean")
    y_select = pn.widgets.Select(name="Y-Axis", options=columns, value="std")

    sidebar = pn.Column("### Controls", x_select, y_select, width=300)

    @pn.depends(x_select.param.value, y_select.param.value)
    def update_plot(x_col: str, y_col: str) -> hv.Points:
        """Update scatter plot based on widget selections."""
        points = hv.Points(df, kdims=[x_col, y_col])
        return points.opts(
            color="#1f77b4",
            size=8,
            responsive=True,
            height=500,
            tools=["hover", "pan", "wheel_zoom", "reset"],
        )

    plot_pane = pn.pane.HoloViews(update_plot, sizing_mode="stretch_both")

    return pn.Row(sidebar, plot_pane, sizing_mode="stretch_width", min_height=550)

@classmethod
def dashboard_layout_hints(cls) -> dict[str, int]:
    """Declare grid space preferences.

    Returns
    -------
    dict[str, int]
        cols: number of GridStack columns to span (1-12).
        rows: number of GridStack rows to span (1+).
    """
    return {"cols": 12, "rows": 3}
```

**Default hints** (from the base class): `{"cols": 6, "rows": 2}`

### Output Format Patterns

#### Parquet Output (MeshStatsFilter / MeshQualityFilter pattern)

```python
import pyarrow as pa
import pyarrow.parquet as pq

def flush(self) -> str | None:
    if not self._rows:
        return None

    new_table = pa.table(
        {col: [row.get(col) for row in self._rows] for col in self._schema.names},
        schema=self._schema,
    )
    self._output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to existing file (worker-level aggregation)
    if self._output_path.exists():
        existing = pq.read_table(str(self._output_path))
        combined = pa.concat_tables([existing, new_table])
        pq.write_table(combined, str(self._output_path))
    else:
        pq.write_table(new_table, str(self._output_path))

    path = str(self._output_path)
    self._rows.clear()
    self._last_artifacts = [path]
    return path
```

#### JSON-Lines Output (MeshInfoFilter pattern)

```python
import json

def __init__(self, output: str | None = None) -> None:
    self._log = get_logger(self)
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

#### Zarr Output (DataArrayStatsFilter pattern)

```python
def flush(self) -> str | None:
    if not self._accumulators:
        return None

    output_path = self._output_path
    output_path.mkdir(parents=True, exist_ok=True)

    for var_name, acc in self._accumulators.items():
        new_stats = acc.finalize()
        group_path = output_path / var_name

        # Merge with existing (worker-level aggregation)
        if group_path.exists():
            existing = xr.open_zarr(str(group_path))
            merged = _merge_moment_datasets([existing, new_stats])
            merged.to_zarr(str(group_path), mode="w", zarr_format=3)
        else:
            new_stats.to_zarr(str(group_path), mode="w", zarr_format=3)

    path = str(output_path)
    self._accumulators.clear()
    self._last_artifacts = [path]
    return path
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
    from tensordict import TensorDictBase

    if not isinstance(td, TensorDictBase):
        return []

    results: list[tuple[str, torch.Tensor]] = []
    for key in td.keys():  # noqa: SIM118
        child = td[key]
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(child, TensorDictBase):
            results.extend(_extract_leaf_tensors(child, prefix=f"{full_key}/"))
        elif isinstance(child, torch.Tensor):
            results.append((full_key, child))
    return results
```

Helper classes for complex state:

```python
class _MyAccumulator:
    """Internal accumulator for running computations.

    Parameters
    ----------
    dims : list[str]
        Dimensions to retain (not reduced).
    """

    def __init__(self, dims: list[str]) -> None:
        self._dims = dims
        self._count: int = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None

    def update(self, data: np.ndarray) -> None:
        """Incorporate a new observation (Welford's online algorithm)."""
        data = np.asarray(data, dtype=np.float64)
        if self._mean is None:
            self._mean = np.zeros_like(data)
            self._m2 = np.zeros_like(data)
        self._count += 1
        delta = data - self._mean
        self._mean += delta / self._count
        delta2 = data - self._mean
        self._m2 += delta * delta2

    def finalize(self) -> dict[str, np.ndarray]:
        """Return final computed values."""
        variance = self._m2 / self._count if self._count > 0 else np.zeros_like(self._mean)
        return {"mean": self._mean, "variance": variance, "count": self._count}
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

Create the test file at `test/domains/<domain>/test_<name>.py`.

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
test pattern in `test/domains/mesh/test_ns_cylinder.py`).

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
        from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource

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

#### Flush and Artifacts Tests

```python
    def test_flush_writes_output(self, tmp_path: pathlib.Path) -> None:
        """Stateful filter writes output on flush."""
        _create_test_mesh(tmp_path / "vtk")
        source = VTKSource.from_path(str(tmp_path / "vtk"))

        output_path = tmp_path / "output.parquet"
        filt = <ClassName>(output=str(output_path))
        list(filt(source[0]))
        result = filt.flush()

        assert result == str(output_path)
        assert output_path.exists()

    def test_artifacts_returns_paths(self, tmp_path: pathlib.Path) -> None:
        """artifacts() returns paths written by flush."""
        output_path = tmp_path / "output.parquet"
        filt = <ClassName>(output=str(output_path))
        # ... process some data ...
        list(filt(gen()))
        filt.flush()

        paths = filt.artifacts()
        assert paths == [str(output_path)]

        # Second call returns empty (consumed)
        assert filt.artifacts() == []

    def test_flush_appends_on_second_call(self, tmp_path: pathlib.Path) -> None:
        """Subsequent flush appends to existing file."""
        output_path = tmp_path / "output.parquet"
        filt = <ClassName>(output=str(output_path))

        # Process first batch
        list(filt(gen_batch_1()))
        filt.flush()

        # Process second batch
        list(filt(gen_batch_2()))
        filt.flush()

        # Verify both batches in output
        table = pq.read_table(str(output_path))
        assert table.num_rows == expected_total_rows
```

#### Merge Tests

```python
    def test_merge_combines_worker_files(self, tmp_path: pathlib.Path) -> None:
        """merge() concatenates per-worker output files."""
        # Create two separate worker files
        worker_0 = tmp_path / "stats_worker_0.parquet"
        worker_1 = tmp_path / "stats_worker_1.parquet"
        # ... write data to each ...

        merged_path = tmp_path / "merged.parquet"
        result = <ClassName>.merge(
            [str(worker_0), str(worker_1)],
            output=str(merged_path),
        )

        assert result == str(merged_path)
        merged = pq.read_table(str(merged_path))
        assert merged.num_rows == expected_combined_rows
```

#### Dashboard Tests

```python
    @pytest.mark.requires("dashboard")
    def test_dashboard_panel_returns_widget(self, tmp_path: pathlib.Path) -> None:
        """dashboard_panel returns a Panel component."""
        # Create an artifact file
        output_path = tmp_path / "stats.parquet"
        # ... write test data ...

        widget = <ClassName>.dashboard_panel([str(output_path)])
        assert widget is not None

    def test_dashboard_panel_no_data(self) -> None:
        """dashboard_panel handles empty artifact list."""
        widget = <ClassName>.dashboard_panel([])
        assert widget is not None  # Returns Markdown message

    def test_dashboard_layout_hints(self) -> None:
        """Layout hints return valid grid dimensions."""
        hints = <ClassName>.dashboard_layout_hints()
        assert "cols" in hints
        assert "rows" in hints
        assert 1 <= hints["cols"] <= 12
        assert hints["rows"] >= 1
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

#### Pipeline Integration Tests

```python
@pytest.mark.e2e
class Test<ClassName>Pipeline:
    """End-to-end pipeline tests."""

    def test_chained_in_pipeline(self, tmp_path: pathlib.Path) -> None:
        """Filter works correctly in a multi-filter pipeline."""
        from physicsnemo_curator.domains.mesh.sinks.mesh_writer import MeshSink
        from physicsnemo_curator.domains.mesh.sources.vtk import VTKSource

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
| `@pytest.mark.requires("atm")` | Skip if atm dependencies not installed |
| `@pytest.mark.requires("dashboard")` | Skip if dashboard deps not installed |
| `@pytest.mark.integration` | Tests that touch filesystem |
| `@pytest.mark.e2e` | End-to-end pipeline tests |
| `@pytest.mark.slow` | Slow tests, excluded from quick CI |

### Running Tests

```bash
# Unit and integration tests (fast, no network)
uv run pytest test/domains/<domain>/test_<name>.py -v

# Pipeline integration test
uv run pytest test/domains/<domain>/test_<name>.py -v -k "Pipeline"

# Full test suite to check for regressions
uv run pytest test/domains/<domain>/ -v -k "not slow"
```

## Step 3: Register the Filter

### Edit the domain `__init__.py`

For **mesh** filters, edit `src/physicsnemo_curator/domains/mesh/__init__.py`:

```python
# Add import (alphabetical order among filters)
from physicsnemo_curator.domains.mesh.filters.<module> import <ClassName>

# Add registration (after existing register_filter calls)
registry.register_filter("mesh", <ClassName>)

# Add to __all__ (alphabetical order)
__all__ = [
    ...,
    "<ClassName>",
    ...,
]
```

For **da** filters, edit `src/physicsnemo_curator/domains/da/__init__.py`.
For **atm** filters, edit `src/physicsnemo_curator/domains/atm/__init__.py`.

### Export any public helpers

If the filter provides a public helper function (like `merge_welford_stats`
in `MeshStatsFilter`), add it to the import and `__all__` as well:

```python
from physicsnemo_curator.domains.mesh.filters.<module> import <ClassName>, <helper_func>

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
  src/physicsnemo_curator/domains/<domain>/filters/<name>.py \
  test/domains/<domain>/test_<name>.py

# Lint
uv run ruff check --fix \
  src/physicsnemo_curator/domains/<domain>/filters/<name>.py \
  test/domains/<domain>/test_<name>.py

# Docstring coverage (must be >= 99%)
uv run interrogate

# Type checking
uv run ty check

# Run filter tests
uv run pytest test/domains/<domain>/test_<name>.py -v

# Run full domain test suite for regressions
uv run pytest test/domains/<domain>/ -v -k "not slow"
```

## Step 5: Commit

Use the `commit` tool (or follow Conventional Commits format):

```text
feat(<domain>): add <ClassName> for <purpose>

<Optional body describing what the filter computes and what output it produces.>
```

## Checklist

Before considering the filter complete, verify:

### Core Implementation
- [ ] SPDX license headers on all new files
- [ ] Filter inherits from `Filter["Mesh"]`, `Filter["xr.DataArray"]`, or `Filter["AtomicData"]`
- [ ] `name` and `description` ClassVars are set
- [ ] `params()` returns all configurable parameters
- [ ] `__init__()` validates parameter constraints
- [ ] `__init__()` uses `self._log = get_logger(self)` for logging
- [ ] `__call__()` receives `Generator[T]`, yields `Generator[T]`
- [ ] Generator is consumed lazily (no upfront `list()` conversion)
- [ ] Pass-through: yields items for downstream consumption

### Stateful Filters (if applicable)
- [ ] `flush()` method returns `str | None`
- [ ] `flush()` appends to existing file (not overwrites) for per-index semantics
- [ ] `flush()` clears accumulated state after writing
- [ ] `_output_path` attribute set as `pathlib.Path` (required for runner integration)
- [ ] `artifacts()` method returns paths written by last flush, then clears
- [ ] `_last_artifacts: list[str]` instance variable for tracking
- [ ] `merge()` static method for combining per-worker output files
- [ ] `merge()` raises `ValueError` on empty input list

### Dashboard (if applicable)
- [ ] `dashboard_panel()` classmethod returns Panel widget or `None`
- [ ] Panel/HoloViews imports inside method body (not at module level)
- [ ] Handles empty/missing artifacts gracefully (returns Markdown message)
- [ ] `dashboard_layout_hints()` classmethod returns `{"cols": N, "rows": M}`
- [ ] Widget uses `@pn.depends` for reactive updates
- [ ] Layout uses `pn.Row(sidebar, plot_pane, sizing_mode="stretch_width")`

### Documentation & Style
- [ ] NumPy-style docstrings on class, `__init__`, `__call__`, `flush`, `artifacts`, `merge`, `dashboard_panel`
- [ ] Helper functions/classes have docstrings and type annotations
- [ ] `from __future__ import annotations` at top
- [ ] `TYPE_CHECKING` block for `Generator` and domain type imports

### Registration
- [ ] Registered in domain `__init__.py` with `registry.register_filter()`
- [ ] Added to `__all__` in domain `__init__.py`
- [ ] Public helpers also exported in `__all__`

### Testing
- [ ] Unit tests: params, name, description
- [ ] Integration tests: pass-through, output files, computed values
- [ ] Flush/artifacts tests (if stateful)
- [ ] Merge tests (if parallel-aware)
- [ ] Dashboard tests (if widget provided)
- [ ] Pipeline test: filter works in a chained pipeline (marked `@pytest.mark.e2e`)
- [ ] Registry test: filter is discoverable

### Quality Gates
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

    def artifacts(self) -> list[str]:
        """Return paths of files produced since the last call.

        Override in stateful filters that write side-effect files.
        The framework calls this after each index to record artifacts.
        Default returns [].
        """
        return []

    @classmethod
    def dashboard_panel(
        cls,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> Any:
        """Return a Panel component visualizing this filter's artifacts.

        Override to provide a custom dashboard widget.
        Default returns None (no widget).
        """
        return None

    @classmethod
    def dashboard_layout_hints(cls) -> dict[str, int]:
        """Declare grid space for dashboard GridStack placement.

        Default: {"cols": 6, "rows": 2}
        """
        return {"cols": 6, "rows": 2}
```

Key differences from `Source[T]`:

- Sources have `__len__` and `__getitem__` -- filters do not
- Filters receive a generator and return a generator (`__call__`)
- Sources are indexed by integer -- filters process streams
- Filters optionally provide `flush()`, `artifacts()`, `merge()` for
  stateful behavior
- Filters optionally provide `dashboard_panel()` for visualization

## Reference: Param Dataclass

From `src/physicsnemo_curator/core/base.py`:

```python
@dataclass(frozen=True)
class Param:
    """Descriptor for a configurable parameter on a pipeline component."""

    name: str           # Must match __init__ keyword argument
    description: str    # Human-readable help text for CLI
    type: type = str    # Expected Python type (str, int, float, bool, pathlib.Path, ...)
    default: Any = REQUIRED  # Use REQUIRED sentinel for mandatory params
    choices: list[str] | None = None  # CLI shows selection prompt if set

    @property
    def required(self) -> bool:
        """Return True if this parameter has no default value."""
        return self.default is REQUIRED
```
