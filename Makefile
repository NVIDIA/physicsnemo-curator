# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: install install-docs setup-ci develop develop-release build \
        format format-check lint lint-check typecheck interrogate \
        test test-core test-mesh test-unit test-integration test-e2e test-device test-rust bench \
        asv-run asv-quick asv-publish asv-preview asv-compare \
        deny docs docs-rust license check clean

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

## Install dev dependencies
install:
	uv sync --group dev

## Install docs dependencies
install-docs:
	uv sync --group docs

## Set up CI environment (install deps + pre-commit hooks)
setup-ci:
	uv sync --group dev && \
	uv run pre-commit install

# ---------------------------------------------------------------------------
# Build (Rust / maturin)
# ---------------------------------------------------------------------------

## Build native extension in dev mode
develop:
	uv run maturin develop

## Build native extension in release mode
develop-release:
	uv run maturin develop --release

## Build wheel
build:
	uv build

# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

## Format Python (ruff) and Rust (cargo fmt)
format:
	uv run ruff format . && \
	cargo fmt --manifest-path src/rust/Cargo.toml

## Check formatting without modifying files
format-check:
	uv run ruff format --check . && \
	cargo fmt --manifest-path src/rust/Cargo.toml -- --check

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

## Lint Python (ruff with auto-fix) and Rust (clippy)
lint:
	uv run ruff check --fix . && \
	cargo clippy --manifest-path src/rust/Cargo.toml -- -D warnings

## Lint without auto-fix
lint-check:
	uv run ruff check . && \
	cargo clippy --manifest-path src/rust/Cargo.toml -- -D warnings

# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------

## Run ty type checker
typecheck:
	uv run ty check

# ---------------------------------------------------------------------------
# Docstring coverage
# ---------------------------------------------------------------------------

## Check docstring coverage with interrogate
interrogate:
	uv run interrogate src/curator

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

## Run Python tests with coverage
test:
	uv run pytest test/ --cov --cov-report=term-missing

## Run only core tests (no optional dependency groups)
test-core:
	uv run pytest test/ -m 'not requires' --cov --cov-report=term-missing

## Run only mesh tests (requires mesh dependency group)
test-mesh:
	uv run pytest test/ -m mesh --cov --cov-report=term-missing

## Run only unit tests (fast, no I/O, no GPU)
test-unit:
	uv run pytest test/ -m unit --cov --cov-report=term-missing

## Run only integration tests (filesystem, network, multi-component)
test-integration:
	uv run pytest test/ -m integration --cov --cov-report=term-missing

## Run only end-to-end pipeline tests
test-e2e:
	uv run pytest test/ -m e2e --cov --cov-report=term-missing

## Run device-parametrised tests (CPU + CUDA when available)
test-device:
	uv run pytest test/ -m device --cov --cov-report=term-missing

## Run Rust tests with nextest
test-rust:
	cargo nextest run --manifest-path src/rust/Cargo.toml

## Run benchmarks (Python + Rust)
bench:
	uv run pytest test/ --benchmark-only && \
	cargo bench --manifest-path src/rust/Cargo.toml

## Run ASV benchmarks on the current HEAD commit
asv-run:
	uv run asv run --quick HEAD^!

## Quick ASV benchmark (single pass, useful for smoke tests)
asv-quick:
	uv run asv run --quick --dry-run HEAD^!

## Build the ASV HTML dashboard from collected results
asv-publish:
	uv run asv publish

## Preview the ASV dashboard locally in a browser
asv-preview:
	uv run asv preview

## Compare ASV results between two revisions (usage: make asv-compare REF1=main REF2=HEAD)
asv-compare:
	uv run asv compare $(REF1) $(REF2)

# ---------------------------------------------------------------------------
# Dependency auditing
# ---------------------------------------------------------------------------

## Run cargo-deny license/advisory checks
deny:
	cargo deny --manifest-path src/rust/Cargo.toml check

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

## Build Sphinx documentation
docs:
	uv run --group docs sphinx-build docs docs/_build/html

## Build Rust documentation (cargo doc)
docs-rust:
	cargo doc --manifest-path src/rust/Cargo.toml --no-deps

# ---------------------------------------------------------------------------
# License
# ---------------------------------------------------------------------------

## Check license headers
license:
	python tests/ci_tests/header_check.py --all-files

# ---------------------------------------------------------------------------
# Aggregate checks
# ---------------------------------------------------------------------------

## Run all checks (format, lint, typecheck, interrogate, deny)
check: format-check lint-check typecheck interrogate deny

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

## Remove build artifacts
clean:
	rm -rf build dist *.egg-info .coverage htmlcov .pytest_cache .ruff_cache .ty_cache
	rm -rf src/rust/target
	rm -rf docs/_build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
