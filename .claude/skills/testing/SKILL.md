---
name: testing
description: >
  Run Python and Rust tests for physicsnemo-curator using uv + pytest with
  coverage reporting via pytest-cov, and cargo-nextest for Rust tests.
  Includes benchmark workflows with pytest-benchmark and criterion.
---

# Testing Skill

## Prerequisites

Ensure dev dependencies are installed:

```bash
make install        # or: uv sync --group dev
make develop        # build native Rust extension: uv run maturin develop
```

## Python Tests

### Run the full test suite with coverage

```bash
uv run pytest test/ --cov --cov-report=term-missing
```

Or use the Makefile shortcut:

```bash
make test
```

### Run a single test file

```bash
uv run pytest test/test_example.py -v
```

### Run a single test function

```bash
uv run pytest test/test_example.py::test_my_function -v
```

### Useful pytest flags

| Flag | Purpose |
|---|---|
| `-v` | Verbose output |
| `-x` | Stop on first failure |
| `-k "expression"` | Run tests matching expression |
| `-m "not slow"` | Skip tests marked as slow |
| `--tb=long` | Full tracebacks |
| `--pdb` | Drop into debugger on failure |
| `-n auto` | Parallel execution (requires pytest-xdist) |

### Coverage reports

```bash
# Terminal report with missing lines (default in make test)
uv run pytest test/ --cov --cov-report=term-missing

# HTML report (opens in browser)
uv run pytest test/ --cov --cov-report=html
# Output: htmlcov/index.html

# XML report (for CI integration)
uv run pytest test/ --cov --cov-report=xml
# Output: coverage.xml

# Combined reports
uv run pytest test/ --cov --cov-report=term-missing --cov-report=html --cov-report=xml
```

Coverage configuration is in `pyproject.toml` under `[tool.coverage.run]` and
`[tool.coverage.report]`. The minimum coverage threshold is set to 80%.

### Python benchmarks

```bash
# Run benchmarks only
uv run pytest test/ --benchmark-only

# Run benchmarks with comparison
uv run pytest test/ --benchmark-compare

# Skip benchmarks during normal test runs
uv run pytest test/ --benchmark-skip
```

Benchmark tests should be marked with `@pytest.mark.benchmark` and use the
`benchmark` fixture from pytest-benchmark.

## Rust Tests

### Run Rust tests with nextest

```bash
cargo nextest run --manifest-path src/rust/Cargo.toml
```

Or use the Makefile shortcut:

```bash
make test-rust
```

### Run specific Rust tests

```bash
cargo nextest run --manifest-path src/rust/Cargo.toml -E "test(parse)"
```

### Rust benchmarks (criterion)

```bash
cargo bench --manifest-path src/rust/Cargo.toml
```

Benchmark results are saved to `src/rust/target/criterion/`. HTML reports are
generated when the `html_reports` feature is enabled.

## Run everything

```bash
make bench    # Python benchmarks + Rust benchmarks
make check    # format-check + lint-check + typecheck + interrogate + deny
```
