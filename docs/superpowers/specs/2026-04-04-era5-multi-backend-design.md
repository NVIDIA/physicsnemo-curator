# ERA5 Multi-Backend Data Source Design

## Goal

Extend `ERA5Source` to support four earth2studio ERA5 backends (ARCO, WB2,
NCAR, CDS) with per-variable automatic fallback routing. Users specify a
priority-ordered list of backends. Each requested variable is assigned to
the highest-priority backend whose lexicon contains it. At fetch time,
variables are grouped by backend, each group is fetched from its
assigned backend, and the results are merged into a single
`xr.DataArray`.

## Background

### Current State

`ERA5Source` (`src/physicsnemo_curator/da/sources/era5.py`, 156 lines) is
hardcoded to `earth2studio.data.ARCO`. It accepts `times`, `variables`,
and `cache`, instantiates `ARCO(cache=cache, verbose=False)`, and calls
`self._arco(time=[t], variable=self._variables)` in `__getitem__`.

### Available earth2studio Backends

| Backend | Class | Variables | Time Res | Special Requirements |
|---------|-------|-----------|----------|---------------------|
| ARCO | `earth2studio.data.ARCO` | 1,408 | 1 hr | None |
| WB2 | `earth2studio.data.WB2ERA5` | 103 | 6 hr | None |
| NCAR | `earth2studio.data.NCAR_ERA5` | 276 | 1 hr | `max_workers` param |
| CDS | `earth2studio.data.CDS` | 117 | 1 hr | CDS API key required |

All four implement the same protocol: `__call__(time=..., variable=...) -> xr.DataArray`
with output dims `(time, variable, lat, lon)` on a 0.25-degree (721x1440)
grid.

84 core variables (surface + 13 pressure levels) are common to all four.
ARCO is the superset. NCAR uniquely has `cp`, `lsp`, `smlt`, `sd`, and
relative humidity at 37 pressure levels. CDS uniquely has `fg10m`, `slor`,
`sdor`, `swvl1`/`swvl2`.

Each backend has a corresponding Lexicon class with a `VOCAB` dict mapping
earth2studio variable IDs to backend-specific names. Variable membership
is checked via `var in LexiconClass`.

## Design

### API Changes to ERA5Source

```python
class ERA5Source(Source["xr.DataArray"]):
    name: ClassVar[str] = "ERA5"
    description: ClassVar[str] = (
        "ERA5 reanalysis via earth2studio (ARCO, WB2, NCAR, CDS)"
    )

    def __init__(
        self,
        times: list[datetime],
        variables: list[str],
        *,
        backend: str | list[str] = "arco",
        backend_options: dict[str, dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
```

**New parameters:**

- `backend`: A single backend name or priority-ordered list. Valid names:
  `"arco"`, `"wb2"`, `"ncar"`, `"cds"`. Default: `"arco"` (backward
  compatible).
- `backend_options`: Optional dict mapping backend names to keyword
  arguments passed to that backend's constructor. Example:
  `{"ncar": {"max_workers": 8}}`. The `cache` and `verbose` parameters
  are set automatically and should not be included here.

**Removed parameters:** None. `cache` is preserved and forwarded to all
backend constructors.

### Backend Registry (Module-Level)

A private mapping from backend name to the earth2studio class and lexicon:

```python
_BACKEND_REGISTRY: dict[str, tuple[str, str, str, str]] = {
    # name: (data_module, data_class, lexicon_module, lexicon_class)
    "arco": ("earth2studio.data", "ARCO", "earth2studio.lexicon", "ARCOLexicon"),
    "wb2":  ("earth2studio.data", "WB2ERA5", "earth2studio.lexicon", "WB2Lexicon"),
    "ncar": ("earth2studio.data", "NCAR_ERA5", "earth2studio.lexicon", "NCAR_ERA5Lexicon"),
    "cds":  ("earth2studio.data", "CDS", "earth2studio.lexicon", "CDSLexicon"),
}
```

Imports are lazy (performed inside `__init__`) to avoid hard dependency on
all backends at module import time. Only backends that are actually needed
(have variables routed to them) get imported and instantiated.

### Per-Variable Routing

At `__init__` time, the source resolves which backend serves each variable:

```python
def _resolve_routing(
    self,
    variables: list[str],
    backend_names: list[str],
) -> dict[str, str]:
    """Map each variable to its highest-priority available backend.

    Returns mapping: variable_name -> backend_name.
    Raises ValueError if any variable is not found in any backend.
    """
    routing: dict[str, str] = {}
    unresolved: list[str] = []
    for var in variables:
        for name in backend_names:
            lexicon = self._get_lexicon(name)
            if var in lexicon:
                routing[var] = name
                break
        else:
            unresolved.append(var)
    if unresolved:
        raise ValueError(
            f"Variables not found in any backend "
            f"({', '.join(backend_names)}): {unresolved}"
        )
    return routing
```

The result is stored as `self._routing: dict[str, str]`.

### Backend Availability

Cloud-native backends (ARCO, WB2, NCAR) are always available when
earth2studio is installed. CDS requires a Copernicus API key.

Availability is checked lazily: if a backend fails to instantiate (e.g.,
CDS credential error), it is skipped with a `warnings.warn()` message and
the next backend in priority order is tried for that variable. If CDS is
the only backend requested and credentials are missing, the error
propagates.

### Fetch & Merge in `__getitem__`

```python
def __getitem__(self, index: int) -> Generator[xr.DataArray]:
    time = self._times[index]

    # Group variables by assigned backend
    groups: dict[str, list[str]] = {}
    for var, backend_name in self._routing.items():
        groups.setdefault(backend_name, []).append(var)

    # Fetch from each backend
    parts: list[xr.DataArray] = []
    for backend_name, var_list in groups.items():
        backend = self._backend_instances[backend_name]
        da = backend(time=[time], variable=var_list)
        parts.append(da)

    # Merge along variable dimension
    if len(parts) == 1:
        result = parts[0]
    else:
        import xarray as xr
        result = xr.concat(parts, dim="variable")

    yield result
```

**Grid alignment safety:** Before concat, assert that lat/lon coordinates
from all parts are identical (within floating-point tolerance). Raise
`ValueError` with a clear message if grids don't align.

### Introspection Properties

```python
@property
def variable_routing(self) -> dict[str, str]:
    """Return mapping of variable name to backend name."""
    return dict(self._routing)

@property
def backends_used(self) -> set[str]:
    """Return set of backend names that have variables routed to them."""
    return set(self._routing.values())

@property
def active_backend(self) -> str | None:
    """Return the single backend name if all variables use one backend.

    Returns None if variables are split across multiple backends.
    """
    backends = self.backends_used
    return next(iter(backends)) if len(backends) == 1 else None
```

### Updated `params()`

```python
@classmethod
def params(cls) -> list[Param]:
    return [
        Param(
            name="times",
            description="Comma-separated ISO timestamps",
            type=str,
        ),
        Param(
            name="variables",
            description="Comma-separated earth2studio variable IDs",
            type=str,
        ),
        Param(
            name="backend",
            description="Backend priority (comma-separated)",
            type=str,
            default="arco",
            choices=["arco", "wb2", "ncar", "cds"],
        ),
        Param(
            name="cache",
            description="Cache downloaded chunks locally",
            type=bool,
            default=True,
        ),
    ]
```

### Registry Name Change

The registered name changes from `"ERA5 (ARCO)"` to `"ERA5"` in
`da/__init__.py`. This reflects the multi-backend capability.

### Backward Compatibility

- Default `backend="arco"` preserves existing behavior exactly.
- `cache` parameter unchanged.
- Return type unchanged: `Generator[xr.DataArray]`.
- When all variables resolve to a single backend, no concat occurs and
  the output is identical to the current implementation.

## Files to Modify

| File | Change |
|------|--------|
| `src/physicsnemo_curator/da/sources/era5.py` | Refactor ERA5Source: add backend param, routing, lazy imports, fetch/merge |
| `src/physicsnemo_curator/da/__init__.py` | Update registry name from `"ERA5 (ARCO)"` to `"ERA5"` |
| `test/da/test_pipeline.py` | Add routing tests, fallback tests, merge tests, backward compat tests; update existing name assertions |
| `docs/user-guide/datasets.md` | Document multi-backend ERA5 usage |

## Testing Strategy

### Unit Tests (mocked backends)

1. **Routing resolution**: Given `backend=["arco", "ncar"]` and variables
   `["t2m", "cp"]`, verify `t2m` routes to `arco` and `cp` routes to `ncar`.

2. **Single-backend fast path**: When all variables exist in first backend,
   no concat occurs.

3. **Multi-backend merge**: When variables split across backends, result
   is concatenated correctly along `variable` dim.

4. **Fallback on unavailability**: CDS backend unavailable (credential
   error), next backend tried, warning emitted.

5. **All-missing error**: Variable not in any backend raises `ValueError`
   with clear message listing tried backends.

6. **Backward compatibility**: `backend="arco"` (default) behaves
   identically to current implementation.

7. **Backend options forwarding**: `backend_options={"ncar": {"max_workers": 8}}`
   is passed to NCAR constructor.

8. **Grid alignment check**: Mismatched lat/lon coordinates raise error.

9. **Variable ordering preserved**: Output DataArray has variables in the
   same order as the input `variables` list regardless of backend grouping.

### Integration Tests (real endpoints, marked slow/e2e)

10. **Multi-backend fetch**: Fetch `t2m` from ARCO + `cp` from NCAR, verify
    merged DataArray shape.

## Non-Goals

- Per-variable time routing (e.g., different time ranges per backend).
- Built-in internal concurrency across backends within a single
  `__getitem__` call (users achieve concurrency by running multiple
  pipelines).
- WB2ERA5 lower-resolution variants (`WB2ERA5_121x240`, `WB2ERA5_32x64`).
  These produce different grid sizes and would break the merge assumption.
