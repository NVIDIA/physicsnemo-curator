# Creating a Custom Filter

This example shows how to implement and register a custom Filter. We create a `LogTransformFilter`
that applies a `log1p` transform to a chosen variable in an xarray DataArray — a common
preprocessing step for ERA5 total precipitation, which has a highly skewed distribution.

## Prerequisites

```bash
uv sync --group da
uv run maturin develop
```

## Usage

```bash
uv run python main.py
```

## How It Works

A filter inherits from `Filter` and implements three things:

1. `name` / `description` class variables (for CLI discovery)
2. `params()` class method (parameter descriptors)
3. `__call__(items)` (the transform logic)

### Step 1 — Define the Filter

```python
from physicsnemo_curator.core.base import Filter, Param

class LogTransformFilter(Filter["xr.DataArray"]):
    name: ClassVar[str] = "Log Transform"
    description: ClassVar[str] = "Apply log1p transform to a DataArray variable"

    @classmethod
    def params(cls) -> list[Param]:
        return [
            Param(name="variable", description="Variable name to apply log1p to", type=str),
        ]

    def __init__(self, variable: str) -> None:
        self._variable = variable

    def __call__(self, items: Generator[xr.DataArray]) -> Generator[xr.DataArray]:
        for da in items:
            if "variable" in da.dims and self._variable in da.coords["variable"].values:
                transformed = da.copy()
                var_idx = list(da.coords["variable"].values).index(self._variable)
                transformed.values[:, var_idx] = np.log1p(da.values[:, var_idx])
                yield transformed
            else:
                yield da
```

### Step 2 — Register the Filter (Optional)

Registration makes the filter discoverable via the global registry and the interactive CLI.
This is optional — unregistered filters work fine in pipelines built with Python code.

```python
from physicsnemo_curator.core.registry import registry

registry.register_filter("da", LogTransformFilter)
```

### Step 3 — Use in a Pipeline

The custom filter plugs into the standard pipeline API just like any built-in filter:

```python
source = ERA5Source(
    times=[datetime(2020, 6, 1, 0), datetime(2020, 6, 1, 6)],
    variables=["tp", "t2m"],
    backend="arco",
)

pipeline = source.filter(LogTransformFilter(variable="tp")).write(
    ZarrSink(output_path="output/extending/log_tp.zarr")
)

results = run_pipeline(pipeline, n_jobs=1, backend="sequential", indices=range(len(pipeline)), progress=True)
```

## Extended API: Stateful Filters

Stateful filters accumulate data across indices and write summary files. They implement two
extra methods beyond `__call__`:

- `flush() -> str | None` — write accumulated state to disk
- `artifacts() -> list[str]` — report files produced by `flush`

They also need:

- `_output_path` attribute — the framework uses this for worker-specific path resolution during
  parallel runs
- `merge(parquet_paths, output)` — `@staticmethod` to combine per-worker shards after parallel
  execution

See `RunningVarianceFilter` in `main.py` for a complete implementation using Welford's online
algorithm.

## Extended API: Dashboard Panels

Filters can provide custom widgets for the metrics dashboard. Implement these class methods:

- `dashboard_panel(artifact_paths, selected_index)` — return a Panel component
- `dashboard_layout_hints() -> dict[str, int]` — declare GridStack space (`cols` and `rows`)

The dashboard discovers these automatically via the widget registry.

## Generator Semantics — Advanced Patterns

`Filter.__call__` receives a `Generator[T]` and must yield `Generator[T]`. This enables several
patterns:

| Pattern | Ratio | Description |
|---------|-------|-------------|
| **Pass-through** | 1:1 | Yield each item unchanged (for analytics) |
| **Transform** | 1:1 | Modify each item |
| **Expand** | 1:N | Yield multiple items per input |
| **Contract/Filter** | N:M (M < N) | Skip items that don't pass checks |
| **Gather** | N:1 | Consume all items, yield a single result |

## Summary

To create a custom filter:

1. Subclass `Filter` with a type parameter (`Filter["xr.DataArray"]`, `Filter["Mesh"]`, etc.)
2. Set `name` and `description` class variables
3. Implement `params()` and `__call__(items)`
4. Optionally register with `registry.register_filter()`

Extended API:

- `flush() -> str | None` — write accumulated state to disk
- `artifacts() -> list[str]` — report side-effect files to PipelineStore
- `merge(paths, output) -> str` — combine per-worker shards (staticmethod)
- `dashboard_panel(artifact_paths, selected_index)` — custom dashboard widget
- `dashboard_layout_hints() -> dict[str, int]` — GridStack placement preferences
