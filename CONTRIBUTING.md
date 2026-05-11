# Contributing to PhysicsNeMo Curator

## Getting Started

1. [Fork](https://help.github.com/en/articles/fork-a-repo) the
   [upstream repository](https://github.com/NVIDIA/physicsnemo-curator).

2. Clone your fork and set up the development environment:

   ```bash
   git clone git@github.com:<you>/physicsnemo-curator.git
   cd physicsnemo-curator

   # Install dev dependencies + build the Rust extension
   uv sync --group dev
   uv run maturin develop

   # Install pre-commit hooks (required for all contributions)
   uv run pre-commit install
   ```

3. Create a branch, make your changes, and open a
   [Pull Request](https://help.github.com/en/articles/creating-a-pull-request)
   against `main`.

## Quality Gates

Every commit is checked by [pre-commit](https://pre-commit.com/) hooks. Run
`uv run pre-commit run --all-files` to validate before pushing. The CI
pipeline enforces the same checks.

| Tool | What it checks | Command |
|---|---|---|
| **ruff** | Python linting + formatting | `uv run ruff check`, `uv run ruff format` |
| **ty** | Python type checking | `uv run ty check` |
| **interrogate** | Docstring coverage (99%) | `uv run interrogate` |
| **cargo fmt / clippy** | Rust formatting + linting | `cargo fmt`, `cargo clippy` |
| **cargo deny** | Rust dependency audit | `cargo deny check` |
| **markdownlint** | Markdown style | via pre-commit |
| **pytest** | Python tests with coverage | `uv run pytest test/ --cov` |
| **cargo nextest** | Rust tests | `cargo nextest run` |

Run all checks at once:

```bash
make check   # format + lint + typecheck + interrogate + deny
make test    # Python tests with coverage
```

## Commit Convention

All commits must follow
[Conventional Commits](https://www.conventionalcommits.org/) format:

```text
<type>(<scope>): <short summary>
```

- **Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`,
  `ci`, `build`, `chore`
- **Scopes** (optional): `mesh`, `da`, `atm`, `core`, `run`, `rust`

## License Headers

All source files (`.py`, `.rs`) must include SPDX headers. The CI lint
job enforces this via
[`.github/scripts/header_check.py`](.github/scripts/header_check.py).

## Sign Your Work

We require [DCO sign-off](https://developercertificate.org/) on all commits:

```bash
git commit -s -m "feat(mesh): add new VTK filter"
```

## Communication

- **GitHub Issues** — bug reports, feature requests
- **GitHub Discussions** — questions, ideas, research directions
