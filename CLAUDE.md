# PhysicsNeMo Curator

Rust-based ETL toolkit for deep learning data curation with Python APIs.

## Architecture

- **Python API**: `src/curator/` — public Python interface
- **Rust core**: `src/rust/` — native extension built with PyO3 + maturin
- **Tests**: `test/` — Python test suite (pytest)
- **Docs**: `docs/` — Sphinx documentation

The Rust extension is exposed as `curator._lib` and imported in
`__init__.py`. Type stubs live at `src/curator/_lib.pyi`.

## Toolchain

| Tool | Purpose | Invocation |
|---|---|---|
| **uv** | Python env & dependency management | `uv sync`, `uv run` |
| **maturin** | Rust/Python build bridge (PyO3) | `uv run maturin develop` |
| **ruff** | Python linting + formatting | `uv run ruff check`, `uv run ruff format` |
| **ty** | Python type checking (Astral) | `uv run ty check` |
| **pytest** | Python testing + coverage | `uv run pytest test/ --cov` |
| **pytest-benchmark** | Python benchmarks | `uv run pytest --benchmark-only` |
| **ASV** | Historical Python benchmarks | `uv run asv run`, `uv run asv publish` |
| **interrogate** | Docstring coverage (99% threshold) | `uv run interrogate` |
| **cargo fmt** | Rust formatting | `cargo fmt --manifest-path src/rust/Cargo.toml` |
| **clippy** | Rust linting | `cargo clippy --manifest-path src/rust/Cargo.toml` |
| **cargo-nextest** | Rust testing | `cargo nextest run --manifest-path src/rust/Cargo.toml` |
| **criterion** | Rust benchmarks | `cargo bench --manifest-path src/rust/Cargo.toml` |
| **cargo-deny** | Rust dependency auditing | `cargo deny --manifest-path src/rust/Cargo.toml check` |

## Quick Reference

```bash
make install            # Install dev dependencies (uv sync --group dev)
make develop            # Build native extension (uv run maturin develop)
make setup-ci           # Install deps + pre-commit hooks

make format             # Format Python (ruff) + Rust (cargo fmt)
make lint               # Lint Python (ruff --fix) + Rust (clippy)
make typecheck          # Run ty type checker
make interrogate        # Check docstring coverage (99%)
make deny               # Audit Rust dependencies

make test               # Python tests with coverage
make test-rust          # Rust tests with nextest
make bench              # Python + Rust benchmarks
make asv-run            # ASV benchmarks on current commit
make asv-publish        # Build ASV HTML dashboard
make asv-preview        # Preview ASV dashboard locally

make check              # All checks (format + lint + typecheck + interrogate + deny)
make docs               # Build Sphinx documentation
make clean              # Remove build artifacts
```

## Code Conventions

- **Python**: Follow ruff defaults. Line length 120. NumPy-style docstrings.
- **Rust**: Follow `rustfmt` defaults. All clippy warnings are errors (`-D warnings`).
- **Docstrings**: 99% coverage enforced by interrogate (excludes test/, docs/).
- **Type checking**: All Python code must pass `ty check`.
- **License**: Apache-2.0. All source files must have SPDX license headers.
- **Tests**: Go in `test/` (Python) or `src/rust/src/` as `#[cfg(test)]` (Rust).

## Dependency Groups

Defined in `pyproject.toml` under `[dependency-groups]`:

- **dev**: ruff, ty, pytest, pytest-cov, pytest-benchmark, asv, pre-commit, interrogate, maturin
- **mesh**: nvidia-physicsnemo, pyvista, pyarrow, torch, warp-lang, s3fs, tqdm, tensordict, jaxtyping, huggingface_hub
- **da**: xarray, earth2studio, zarr, gcsfs
- **docs**: sphinx, nvidia-sphinx-theme, myst-parser, sphinx-autoapi, sphinx-autodoc-typehints, etc.

Optional extras in `[project.optional-dependencies]`:

- **parallel**: joblib, dask (for `run_pipeline` multiprocess backends)
- **cli**: click, questionary (for interactive CLI)

Install with: `uv sync --group dev` or `uv sync --group docs`

## Custom Tools Available

Claude tools in `.claude/tools/`:

| Tool | Description |
|---|---|
| `ruff` | Run ruff linter (auto-fix by default) |
| `ruff-format` | Run ruff formatter |
| `uv` | General-purpose uv command runner |
| `ty` | Run ty type checker |
| `cargo-fmt` | Run cargo fmt on Rust code |
| `cargo-clippy` | Run clippy on Rust code |
| `cargo-nextest` | Run Rust tests with nextest |
| `cargo-deny` | Audit Rust dependencies |
| `asv` | Run ASV historical benchmarks |

## Skills Available

- **testing**: Comprehensive guide for running Python and Rust tests, coverage, and benchmarks.
