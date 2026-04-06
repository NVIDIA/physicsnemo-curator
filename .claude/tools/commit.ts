import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description: `Stage files and create a git commit, always running pre-commit hooks first.

Commit message MUST follow Conventional Commits format:
  <type>(<scope>): <short summary>

  [optional body]

Types: feat, fix, refactor, test, docs, style, perf, ci, build, chore
Scope: optional, e.g. mesh, da, core, run, cli, rust
Summary: imperative mood, lowercase, no period, max 72 chars

Examples:
  feat(mesh): add NavierStokesCylinderSource for Parquet datasets
  fix: configure ty type checker for Python 3.12
  test(da): add fetch/merge tests for multi-backend ERA5Source
  refactor(core): simplify FileStore caching logic
  docs: update datasets guide with new source examples`,
  args: z.object({
    message: z
      .string()
      .describe(
        "Commit message in Conventional Commits format: <type>(<scope>): <summary>"
      ),
    paths: z
      .string()
      .default(".")
      .describe(
        'Files to stage (space-separated). Use "." to stage all changes'
      ),
    amend: z
      .boolean()
      .default(false)
      .describe("Amend the previous commit instead of creating a new one"),
  }),
  async execute({ message, paths, amend }) {
    const cwd = import.meta.dir + "/../..";

    // Validate conventional commit format
    const pattern =
      /^(feat|fix|refactor|test|docs|style|perf|ci|build|chore)(\([a-z0-9_-]+\))?!?: .+/;
    const firstLine = message.split("\n")[0];
    if (!pattern.test(firstLine)) {
      return `ERROR: Commit message does not follow Conventional Commits format.

Expected: <type>(<scope>): <summary>
Got:      ${firstLine}

Valid types: feat, fix, refactor, test, docs, style, perf, ci, build, chore
Scope is optional, e.g. (mesh), (da), (core)
Summary should be imperative mood, lowercase, no trailing period.`;
    }

    if (firstLine.length > 72) {
      return `ERROR: Commit summary line exceeds 72 characters (${firstLine.length}).

Shorten the first line and move details to the body:
  <type>(<scope>): <short summary>

  <detailed explanation>`;
    }

    // Stage files
    const addResult = Bun.spawnSync(["git", "add", ...paths.split(/\s+/)], {
      cwd,
    });
    if (addResult.exitCode !== 0) {
      return `ERROR staging files:\n${addResult.stderr.toString()}`;
    }

    // Check if there are staged changes
    const diffResult = Bun.spawnSync(
      ["git", "diff", "--cached", "--quiet"],
      { cwd }
    );
    if (!amend && diffResult.exitCode === 0) {
      return "ERROR: No staged changes to commit. Stage files first or check paths.";
    }

    // Run pre-commit on staged files
    const precommitResult = Bun.spawnSync(
      ["uv", "run", "pre-commit", "run"],
      { cwd, timeout: 300_000 }
    );
    const precommitOut = precommitResult.stdout.toString();
    const precommitErr = precommitResult.stderr.toString();

    if (precommitResult.exitCode !== 0) {
      // Re-stage files that pre-commit may have auto-fixed
      Bun.spawnSync(["git", "add", ...paths.split(/\s+/)], { cwd });

      // Retry pre-commit after auto-fixes
      const retryResult = Bun.spawnSync(
        ["uv", "run", "pre-commit", "run"],
        { cwd, timeout: 300_000 }
      );
      const retryOut = retryResult.stdout.toString();

      if (retryResult.exitCode !== 0) {
        return `PRE-COMMIT FAILED (after auto-fix retry):\n${retryOut}\n${retryResult.stderr.toString()}\n\nFix the issues above and try again.`;
      }
    }

    // Create the commit
    const commitArgs = ["git", "commit", "-m", message];
    if (amend) commitArgs.push("--amend");
    const commitResult = Bun.spawnSync(commitArgs, { cwd });
    const commitOut = commitResult.stdout.toString();
    const commitErr = commitResult.stderr.toString();

    if (commitResult.exitCode !== 0) {
      return `ERROR creating commit:\n${commitOut}\n${commitErr}`;
    }

    // Show result
    const logResult = Bun.spawnSync(
      ["git", "log", "--oneline", "-1"],
      { cwd }
    );

    return `Commit created successfully.\n${logResult.stdout.toString().trim()}\n\nPre-commit: ${precommitOut.includes("Passed") || precommitResult.exitCode === 0 ? "all hooks passed" : "passed after auto-fix"}`;
  },
});
