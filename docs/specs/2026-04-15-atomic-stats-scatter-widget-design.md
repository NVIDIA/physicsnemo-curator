# AtomicStatsScatterWidget Design Spec

**Date:** 2026-04-15
**Status:** Approved

## Overview

Add an interactive scatter plot widget to the PhysicsNeMo Curator dashboard for
visualizing `AtomicStatsFilter` parquet output. Users can explore relationships
between computed statistics (mean, std, variance, etc.) across different atomic
data fields.

## Requirements

### Functional Requirements

1. **Interactive Scatter Plot**
   - X and Y axes selectable from available statistics columns
   - Color points by user-selected categorical dimension
   - Filter points by level (node/edge/system/extra)

2. **Hover Tooltip** displays:
   - Pipeline index
   - Field metadata: `field_key`, `level`, `component`
   - All statistical values: mean, std, var, min, max, median, abs_mean, abs_max, skewness, kurtosis
   - Count metadata: n_values, n_components

3. **Axis Options** (all statistics from parquet schema):
   - Core: `mean`, `std`, `var`, `min`, `max`, `median`
   - Absolute: `abs_mean`, `abs_max`
   - Higher moments: `skewness`, `kurtosis`
   - Counts: `n_values`, `n_components`

4. **Color-by Options**:
   - `level` (node/edge/system/extra)
   - `field_key` (positions, forces, energies, etc.)
   - `component` (0, 1, 2, -1 for scalars)

5. **Data Scope**: Use existing dashboard artifact filtering mechanism

### Non-Functional Requirements

- Follow existing `WidgetProvider` protocol
- Use Holoviews + Bokeh (matches existing dashboard stack)
- Responsive layout with sidebar controls

## Architecture

### File Structure

```text
src/physicsnemo_curator/dashboard/widgets/
├── __init__.py      # Edit: register AtomicStatsScatterWidget
├── base.py          # Existing: WidgetProvider protocol
├── mesh.py          # Existing: MeanFilterWidget (reference)
└── atm.py           # NEW: AtomicStatsScatterWidget
```

### Class Design

```python
# src/physicsnemo_curator/dashboard/widgets/atm.py

class AtomicStatsScatterWidget:
    """Interactive scatter plot for AtomicStatsFilter parquet output."""

    name: str = "Atomic Statistics Scatter"
    filter_name: str = "AtomicStatsFilter"

    def panel(
        self,
        artifact_paths: list[str],
        selected_index: int | None = None,
    ) -> pn.viewable.Viewable:
        """Return Panel component with sidebar + scatter plot."""
```

### UI Layout (Design B - Sidebar)

```text
┌──────────────────────────────────────────────────────────┐
│  ┌─────────────┐  ┌────────────────────────────────────┐ │
│  │  X-Axis     │  │                                    │ │
│  │  [dropdown] │  │                                    │ │
│  │             │  │                                    │ │
│  │  Y-Axis     │  │         Scatter Plot               │ │
│  │  [dropdown] │  │         (Holoviews/Bokeh)          │ │
│  │             │  │                                    │ │
│  │  Color by   │  │         • Hover shows tooltip      │ │
│  │  [dropdown] │  │         • Pan/zoom tools           │ │
│  │             │  │         • Legend for colors        │ │
│  │  Filter:    │  │                                    │ │
│  │  ☑ node     │  │                                    │ │
│  │  ☑ edge     │  │                                    │ │
│  │  ☑ system   │  │                                    │ │
│  │  ☑ extra    │  └────────────────────────────────────┘ │
│  └─────────────┘                                         │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

1. `panel()` receives `artifact_paths` (list of parquet files)
2. Load and concatenate parquet files into single DataFrame
3. Create Panel widgets for controls (Select, CheckBoxGroup)
4. Create Holoviews Points plot with Bokeh backend
5. Wire up `param.watch` to update plot when controls change
6. Return `pn.Row(sidebar, plot)` layout

### Parquet Schema Reference

The `AtomicStatsFilter` outputs parquet with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `field_key` | string | Field name (e.g., "positions", "forces") |
| `level` | string | "node", "edge", "system", or "extra" |
| `component` | int32 | -1 for scalars, 0/1/2/... for vector components |
| `n_values` | int64 | Number of values seen |
| `n_components` | int32 | 1 for scalar, N for N-dim vector |
| `mean` | float64 | Arithmetic mean |
| `std` | float64 | Standard deviation |
| `var` | float64 | Variance |
| `min` | float64 | Minimum value |
| `max` | float64 | Maximum value |
| `median` | float64 | Median value |
| `abs_mean` | float64 | Mean of absolute values |
| `abs_max` | float64 | Maximum absolute value |
| `skewness` | float64 | Fisher-Pearson coefficient |
| `kurtosis` | float64 | Excess kurtosis |

## Implementation Plan

### Task 1: Create widget file

Create `src/physicsnemo_curator/dashboard/widgets/atm.py`:

- Import dependencies (panel, holoviews, pandas)
- Define `STAT_COLUMNS` and `COLOR_BY_OPTIONS` constants
- Implement `AtomicStatsScatterWidget` class
- Implement `_load_data()` helper to read/concat parquet files
- Implement `_create_sidebar()` with Panel widgets
- Implement `_create_plot()` with Holoviews Points
- Implement `panel()` method returning pn.Row layout

### Task 2: Register widget

Edit `src/physicsnemo_curator/dashboard/widgets/__init__.py`:

- Add try/except import for `AtomicStatsScatterWidget`
- Register in `WidgetRegistry.__init__`

### Task 3: Add tests

Create `test/dashboard/test_atm_widget.py`:

- Test widget instantiation
- Test `panel()` returns valid Panel object
- Test with mock parquet data

### Task 4: Update docstrings

Ensure 99% docstring coverage per project requirements.

## Testing Strategy

1. **Unit tests**: Widget instantiation, data loading, control creation
2. **Integration test**: Full `panel()` call with sample parquet
3. **Manual verification**: Run dashboard with real `AtomicStatsFilter` output

## Dependencies

Already in `dashboard` optional extra:

- `panel>=1.3`
- `holoviews>=1.18`
- `bokeh>=3.3`
- `pandas>=2.0`
- `pyarrow>=14.0`

## Acceptance Criteria

- [ ] Widget appears in dashboard Pipeline tab for AtomicStatsFilter artifacts
- [ ] Scatter plot renders with selectable X/Y axes
- [ ] Color-by dropdown changes point colors
- [ ] Level checkboxes filter displayed points
- [ ] Hover tooltip shows all required information
- [ ] All tests pass
- [ ] Docstring coverage maintained at 99%
