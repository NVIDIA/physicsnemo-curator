<!---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
-->

# Using AI Agents

PhysicsNeMo Curator ships with **Claude skills** — structured instructions that
guide an AI coding agent through implementing new components.  If you use
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) (or a compatible
agent), these skills automate much of the boilerplate and ensure new code
follows project conventions from the start.

## Available Skills

Five skills live in `.claude/skills/`:

| Skill | Trigger | What it does |
|-------|---------|--------------|
| **add-source** | "Add a new data source" | Generates source class with params, tests, registry |
| **add-filter** | "Add a new filter" | Generates filter with correct generator patterns and tests |
| **add-sink** | "Add a new sink" | Generates sink with naming strategy, append logic, and tests |
| **curator-reviewer** | "Review this PR" | Reviews PRs against project standards (8-pass checklist) |
| **testing** | "Run tests" | Guides running Python/Rust tests with correct tooling |

## How to Use Them

1. Open the repository in Claude Code (or any agent that supports Claude
   skills).
2. Describe what you want to build — for example:

   > *"Add a new source that reads Parquet files from S3 and yields Mesh
   > objects."*

3. The agent detects the matching skill and follows its guided workflow:
   - Asks discovery questions (file format, domain, naming, etc.)
   - Generates the implementation with correct patterns
   - Creates tests, registers the component, and runs quality checks

## What the Skills Automate

Each skill encodes the project's conventions so you don't have to remember
them all:

- SPDX license headers
- `from __future__ import annotations` and `TYPE_CHECKING` import patterns
- `ClassVar` declarations for `name` and `description`
- `params()` classmethod with proper `Param` objects
- NumPy-style docstrings (99 % coverage requirement)
- Generator semantics (sources and filters)
- Test scaffolding with `pytest.mark.requires`
- Registry registration in `__init__.py`
- Conventional Commit messages

## Without an Agent

The skills are plain Markdown files — you can read them directly for reference
even if you're not using an AI agent:

```bash
cat .claude/skills/add-source/SKILL.md
cat .claude/skills/add-filter/SKILL.md
cat .claude/skills/add-sink/SKILL.md
cat .claude/skills/curator-reviewer/SKILL.md
cat .claude/skills/testing/SKILL.md
```

They serve as detailed checklists for implementing components by hand.
