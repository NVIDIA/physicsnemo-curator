---
name: curator-reviewer
description: >
  Review PRs against PhysicsNeMo Curator standards. Runs 8 review passes
  covering API conformance, correctness, licensing, quality gates, test
  coverage, performance, code quality, and style consistency. Produces a
  prioritized report (P0/P1/P2/NIT) and optionally posts review comments
  to the PR.
---

# Curator PR Reviewer

This skill provides a structured, opinionated code review workflow for
PhysicsNeMo Curator pull requests. It checks changed code against the
project's API conventions, quality gates, and style patterns, then
produces a prioritized findings report.

**Report-only** — no automatic fixes are applied. The reviewer presents
findings and the user decides what to act on.

## When to Use

- Reviewing any PR to `NVIDIA/physicsnemo-curator`
- Self-reviewing your own work before requesting external review
- After Greptile review to catch project-specific issues it may miss

## Priority Levels

| Priority | Label | Meaning | Action required |
|----------|-------|---------|-----------------|
| **P0** | Must Fix | Blocks merge. API violations, broken contracts, missing SPDX. | Always |
| **P1** | Should Fix | Missing test coverage, performance problems, resource leaks. | Before merge |
| **P2** | Consider | Over-commenting, verbose code, naming inconsistencies. | Discretion |
| **NIT** | Stylistic | Differs from codebase conventions but not wrong. | Optional |

## Workflow

Follow these steps in order. Use the TodoWrite tool to track each pass.

### Step 0: Fetch the PR

Identify the PR to review. Accept a PR number, URL, or auto-detect from
the current branch.

```bash
# Fetch PR metadata and diff
gh pr view <number> --repo NVIDIA/physicsnemo-curator
gh pr diff <number> --repo NVIDIA/physicsnemo-curator
```

Record:

- PR number, title, author, branch
- List of changed files (additions, modifications, deletions)
- Total lines added / removed

Read each changed file **in full** (not just the diff) — context matters.
For modified files, also read the base version to understand what changed.

### Step 1: API Conformance (P0)

Check every new or modified Source, Filter, or Sink against the ABCs in
`src/physicsnemo_curator/core/base.py`.

**Source[T] checklist:**

- [ ] `name: ClassVar[str]` declared
- [ ] `description: ClassVar[str]` declared
- [ ] `params()` returns `list[Param]` (classmethod)
- [ ] `__len__()` returns `int`
- [ ] `__getitem__(index: int)` returns `Generator[T]`
- [ ] `__getitem__` raises `IndexError` for out-of-range
- [ ] Generator semantics: yields items lazily, no eager loading of all data
- [ ] File discovery happens in `__init__`, not `__getitem__`

**Filter[T] checklist:**

- [ ] `name`, `description`, `params()` present
- [ ] `__call__(items: Generator[T]) -> Generator[T]`
- [ ] Generator contract: iterates input lazily, yields results
- [ ] If stateful: `flush() -> str | None` implemented
- [ ] Cardinality documented (1:1, 1:N, N:fewer)

**Sink[T] checklist:**

- [ ] `name`, `description`, `params()` present
- [ ] `__call__(items: Iterator[T], index: int) -> list[str]`
- [ ] Returns list of written file paths
- [ ] Consumes iterator (does not convert to list unless necessary)

**Pipeline checklist:**

- [ ] Components are registered in the domain `__init__.py`
- [ ] Pipeline builder chain works: `source.filter(...).write(...)`

Flag any deviation as **P0**.

### Step 2: Correctness & Safety (P0)

Check for:

- [ ] Missing error handling on I/O operations (file open, network calls)
- [ ] Broken generator contracts (returning instead of yielding, not
      iterating the input generator in a filter)
- [ ] Incorrect indexing (off-by-one, missing `IndexError` on sources)
- [ ] Resource leaks (unclosed files, sockets, database connections)
- [ ] Race conditions in parallel-safe code
- [ ] Mutable default arguments
- [ ] Bare `except:` clauses (should catch specific exceptions)
- [ ] Silent data loss (swallowing exceptions, skipping items without logging)

Flag any issue as **P0**.

### Step 3: License & Headers (P0)

Every new file must start with:

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

For Rust files, use `//` comment syntax with the same text.

Check:

- [ ] All new files have the SPDX header
- [ ] Header matches the exact format above (not a paraphrase)
- [ ] Commit messages follow Conventional Commits format

Flag missing headers as **P0**.

### Step 4: Quality Gates (P0)

Run the automated quality checks against changed files:

```bash
# Lint changed Python files
uv run ruff check <changed_files>

# Type-check (full project — ty doesn't support file-level checking)
uv run ty check --exclude 'examples-old/**' --exclude 'benchmarks/**'

# Docstring coverage
uv run interrogate
```

Check:

- [ ] No ruff errors or warnings on changed files
- [ ] No new ty errors introduced by the PR
- [ ] Interrogate still at 99%+ coverage
- [ ] For Rust changes: `cargo clippy` and `cargo fmt --check` clean

Any quality gate failure is **P0**.

### Step 5: Test Coverage (P1)

Evaluate testing adequacy:

- [ ] Every new public class/function has at least one test
- [ ] Test file exists in the correct location (`test/<domain>/`)
- [ ] Correct markers applied:
  - `@pytest.mark.requires("mesh")` or `@pytest.mark.requires("da")`
  - `@pytest.mark.e2e` for end-to-end tests
  - `@pytest.mark.slow` for tests > 10s
- [ ] Edge cases covered (empty input, single item, error conditions)
- [ ] Stateful filters: flush tested, multi-batch accumulation tested
- [ ] Sinks: write, roundtrip, append (if applicable) tested
- [ ] Sources: `__len__`, `__getitem__`, `IndexError`, generator semantics tested

Run coverage on affected test files if feasible:

```bash
uv run pytest test/<domain>/<test_file> --cov=src/physicsnemo_curator/<module> --cov-report=term-missing
```

Missing tests for public API surfaces are **P1**. Missing edge case tests
are **P2**.

### Step 6: Performance & Profiling (P1)

Look for performance anti-patterns:

- [ ] **Eager loading**: `list(generator)` or `[x for x in gen]` where
      lazy iteration would work — Sources should yield lazily, filters
      should not materialize the full stream
- [ ] **Unbounded memory accumulation**: Growing lists/dicts without
      bounds in stateful filters (acceptable if flush() clears them)
- [ ] **Unnecessary copies**: `.clone()`, `.copy()`, `deepcopy()` where
      in-place mutation is safe
- [ ] **Repeated I/O**: Reading the same file multiple times when once
      would suffice
- [ ] **Missing streaming**: Loading entire datasets into memory instead
      of streaming through the pipeline
- [ ] **Quadratic patterns**: Nested loops over data that could be
      O(n) with better data structures
- [ ] **Process isolation awareness**: Stateful filter state is NOT
      merged back when using `run_pipeline(n_jobs>1)`. The merge()
      pattern must be used post-hoc. Check that new stateful filters
      document this.

Flag performance issues as **P1**. Profiling-specific concerns
(missing `ProfiledPipeline` support, metric collection gaps) are **P2**.

### Step 7: Code Quality (P2)

Review for clean, concise implementation:

- [ ] **Over-commenting**: Comments that restate what the code does.
      Code should be self-documenting. Comments should explain *why*,
      not *what*. Bad: `# Increment the counter` above `counter += 1`.
      Good: `# Reset after flush to avoid double-counting`.
- [ ] **Verbose implementations**: Could the same logic be expressed
      more concisely without sacrificing readability?
- [ ] **Dead code**: Unused imports, unreachable branches, commented-out
      code blocks
- [ ] **Naming**: Do names follow existing patterns in the codebase?
  - Sources: `<Name>Source` (e.g. `VTKSource`, `ERA5Source`)
  - Filters: `<Name>Filter` (e.g. `MeanFilter`, `MomentsFilter`)
  - Sinks: `<Name>Sink` (e.g. `MeshSink`, `ZarrSink`)
  - Test classes: `Test<ClassName>` (e.g. `TestMeanFilter`)
  - Test files: `test_<module>.py`
- [ ] **Docstring quality**: NumPy-style, with Parameters / Returns /
      Examples sections. Not overly verbose — each param gets one line
      of description.
- [ ] **Function length**: Functions over ~50 lines should be considered
      for decomposition. Not a hard rule, but flag for discussion.

### Step 8: Style Consistency (NIT)

Compare against existing patterns in the package. These are not
correctness issues — they're about matching what's already there.

- [ ] **Import ordering**: stdlib → third-party → local, with
      `TYPE_CHECKING` block for type-only imports
- [ ] **`from __future__ import annotations`**: Present in all source
      files (required for `ClassVar[str]` without quotes)
- [ ] **Docstring phrasing**: Existing codebase uses imperative mood
      for method docstrings ("Return the ...", "Compute the ...")
- [ ] **Type annotations**: All public function signatures have type
      hints. Use `Generator[T]`, `Iterator[T]`, not bare `generator`.
- [ ] **Test fixture patterns**: `tmp_path` for temporary directories,
      helper functions for creating test data, domain markers
- [ ] **Parameter style**: `Param(name=..., description=..., type=...,
      default=...)` — check that new params match existing conventions

## Output Format

After completing all passes, compile findings into a structured report.

```markdown
## PR Review: #<number> — <title>

**Author:** <author> | **Branch:** <branch> | **Files changed:** <count>

### Summary
<1-3 sentence overall assessment. Is this ready to merge, close to
ready, or needs significant work?>

### Findings

#### P0 — Must Fix
| # | File:Line | Pass | Finding | Suggestion |
|---|-----------|------|---------|------------|
| 1 | `src/.../foo.py:42` | API | Missing `name` ClassVar | Add `name: ClassVar[str] = "..."` |

#### P1 — Should Fix
| # | File:Line | Pass | Finding | Suggestion |
|---|-----------|------|---------|------------|

#### P2 — Consider
| # | File:Line | Pass | Finding | Suggestion |
|---|-----------|------|---------|------------|

#### NIT — Stylistic
| # | File:Line | Pass | Finding | Suggestion |
|---|-----------|------|---------|------------|

### Test Coverage Assessment
<What's tested, what's missing, coverage % if measured>

### Performance Notes
<Any profiling concerns, streaming issues, or memory patterns>
```

If there are zero findings for a priority level, omit that section.

## Step 9: Post to PR (Optional)

After presenting the report, ask the user:

> "Would you like me to post these findings as review comments on
> PR #\<number\>?"

If the user says **yes**, post the findings using `gh api` as a single
pull request review with line-level comments.

### Comment Template

Each comment follows this format:

```markdown
**[{PRIORITY}]** {TITLE}

{DESCRIPTION}

```suggestion
{SUGGESTED_CODE_FIX}
```​
```

- `{PRIORITY}`: One of `P0`, `P1`, `P2`, `NIT`
- `{TITLE}`: Short (< 10 words) summary of the issue
- `{DESCRIPTION}`: 1-3 sentences explaining the problem and why it
  matters. Reference existing code patterns where relevant.
- `{SUGGESTED_CODE_FIX}`: Optional. If the fix is clear, include it in
  a GitHub suggestion block so the author can apply it with one click.
  Omit the suggestion block if the fix requires broader refactoring.

### Posting via `gh api`

Submit all comments as a single review (not individual comments):

```bash
gh api repos/NVIDIA/physicsnemo-curator/pulls/<number>/reviews \
  --method POST \
  --field event="COMMENT" \
  --field body="<overall summary>" \
  --field 'comments=[
    {
      "path": "<file>",
      "line": <line_number>,
      "body": "<formatted comment>"
    }
  ]'
```

Use `event="COMMENT"` (not `REQUEST_CHANGES` or `APPROVE`) — the
reviewer reports findings but the merge decision is the user's.

For findings that apply to a file but not a specific line (e.g. missing
SPDX header, missing test file), use line 1 of the relevant file, or
include them only in the review body summary.

### Review Body Template

The top-level review body summarizes the full review:

```markdown
## Curator Review: <count> finding(s)

**P0:** <n> | **P1:** <n> | **P2:** <n> | **NIT:** <n>

<1-2 sentence summary of overall assessment>
```

## Checklist

Use this checklist to verify the review is complete:

- [ ] PR diff fetched and all changed files read in full
- [ ] Pass 1: API conformance checked for all new/modified components
- [ ] Pass 2: Correctness & safety reviewed
- [ ] Pass 3: SPDX headers verified on all new files
- [ ] Pass 4: Quality gates run (ruff, ty, interrogate)
- [ ] Pass 5: Test coverage assessed
- [ ] Pass 6: Performance patterns reviewed
- [ ] Pass 7: Code quality evaluated (comments, verbosity, naming)
- [ ] Pass 8: Style consistency checked against existing patterns
- [ ] Findings compiled into structured report
- [ ] Priority levels assigned correctly (P0/P1/P2/NIT)
- [ ] User asked about posting to PR

## Reference: Project Standards

Quick reference for the standards checked by this review.

### Quality Thresholds

| Check | Threshold | Tool |
|-------|-----------|------|
| Python lint | Zero errors | `ruff check` (rules: E, F, W, I, D, UP, N, B, A, C4, SIM, TCH, PTH, ERA) |
| Python format | ruff defaults | `ruff format` (line length 120, double quotes) |
| Type checking | Zero errors | `ty check` (Python 3.12) |
| Docstring coverage | 99% | `interrogate` (excludes test/, docs/) |
| Test coverage | 80% minimum | `pytest-cov` |
| Rust lint | Zero warnings | `clippy -D warnings` |
| Rust format | rustfmt defaults | `cargo fmt` |
| License | SPDX on all files | Manual check |
| Commits | Conventional Commits | `<type>(<scope>): <summary>` |

### API Contracts (from `core/base.py`)

```text
Source[T]:
  name: ClassVar[str]
  description: ClassVar[str]
  params() -> list[Param]
  __len__() -> int
  __getitem__(index: int) -> Generator[T]

Filter[T]:
  name: ClassVar[str]
  description: ClassVar[str]
  params() -> list[Param]
  __call__(items: Generator[T]) -> Generator[T]
  flush() -> str | None  (optional, for stateful filters)

Sink[T]:
  name: ClassVar[str]
  description: ClassVar[str]
  params() -> list[Param]
  __call__(items: Iterator[T], index: int) -> list[str]
```

### Naming Conventions

| Component | Class name | File name | Test class | Test file |
|-----------|-----------|-----------|------------|-----------|
| Source | `<Name>Source` | `<name>.py` | `Test<Name>Source` | `test_<name>.py` |
| Filter | `<Name>Filter` | `<name>.py` | `Test<Name>Filter` | `test_<name>.py` |
| Sink | `<Name>Sink` | `<name>_writer.py` | `Test<Name>Sink` | `test_<name>.py` |

### Test Markers

```python
@pytest.mark.requires("mesh")   # skip if mesh deps missing
@pytest.mark.requires("da")     # skip if da deps missing
@pytest.mark.e2e                # end-to-end pipeline test
@pytest.mark.slow               # excluded by default
@pytest.mark.benchmark          # performance benchmark
```
