<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Style Guide

This page summarises the key coding conventions for PhysicsNeMo Curator.
The authoritative reference is **`CLAUDE.md`** in the repository root — consult
it for the full details.

## License Headers

Every source file must include an SPDX license header:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
```

For Markdown and HTML files use comment syntax appropriate to the format.

## Python Conventions

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/) with a line
  length of 120.
- **Type checking**: All code must pass `ty check`.
- **Docstrings**: NumPy-style.  99 % coverage enforced by
  [interrogate](https://interrogate.readthedocs.io/) (excludes `test/` and
  `docs/`).
- **Imports**: Use `from __future__ import annotations` and gate heavy imports
  behind `TYPE_CHECKING`.

## Rust Conventions

- **Formatter**: `rustfmt` defaults via `cargo fmt`.
- **Linter**: clippy with all warnings treated as errors (`-D warnings`).

## Commit Messages

All commits use [Conventional Commits](https://www.conventionalcommits.org/)
format:

```text
<type>(<scope>): <short summary>
```

- **Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `ci`,
  `build`, `chore`
- **Scopes** (optional): `mesh`, `da`, `core`, `run`, `cli`, `rust`
- Summary in imperative mood, lowercase, no trailing period, max 72 chars.

Pre-commit hooks enforce formatting and linting automatically.

## Toolchain Quick Reference

| Tool | Command | Purpose |
|------|---------|---------|
| ruff | `uv run ruff check` / `uv run ruff format` | Python lint + format |
| ty | `uv run ty check` | Python type checking |
| interrogate | `uv run interrogate` | Docstring coverage |
| cargo fmt | `cargo fmt --manifest-path src/rust/Cargo.toml` | Rust formatting |
| clippy | `cargo clippy --manifest-path src/rust/Cargo.toml` | Rust linting |
| cargo-deny | `cargo deny --manifest-path src/rust/Cargo.toml check` | Dependency audit |

Run all checks at once with:

```bash
make check   # format + lint + typecheck + interrogate + deny
```
