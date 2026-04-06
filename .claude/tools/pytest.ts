import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description: `Run Python tests with pytest and optional coverage reporting.

Supports filtering by test category (unit, integration, e2e), dependency
group (mesh, da, core), specific files, and arbitrary pytest expressions.

Examples:
  scope: "unit"               → fast tests, no I/O or GPU
  scope: "integration"        → filesystem/network tests
  scope: "e2e"                → end-to-end pipeline tests
  scope: "all"                → everything (default)
  group: "mesh"               → only mesh-group tests
  group: "da"                 → only da-group tests
  group: "core"               → tests without optional deps
  path: "test/mesh/test_ns_cylinder.py"  → single file
  filter: "TestStatsFilter"   → pytest -k expression`,
  args: z.object({
    scope: z
      .enum(["all", "unit", "integration", "e2e"])
      .default("all")
      .describe("Test category to run"),
    group: z
      .enum(["all", "mesh", "da", "core"])
      .default("all")
      .describe("Dependency group to filter by"),
    path: z
      .string()
      .default("test/")
      .describe("Test file or directory path"),
    filter: z
      .string()
      .optional()
      .describe("pytest -k expression to filter test names"),
    coverage: z
      .boolean()
      .default(true)
      .describe("Enable coverage reporting"),
    verbose: z
      .boolean()
      .default(true)
      .describe("Verbose output (-v flag)"),
    exitfirst: z
      .boolean()
      .default(false)
      .describe("Stop on first failure (-x flag)"),
    slow: z
      .boolean()
      .default(false)
      .describe("Include tests marked @pytest.mark.slow"),
  }),
  async execute({ scope, group, path, filter, coverage, verbose, exitfirst, slow }) {
    const cwd = import.meta.dir + "/../..";
    const args = ["uv", "run", "pytest", path];

    // Build marker expression parts
    const markers: string[] = [];

    // Scope filter
    if (scope !== "all") {
      markers.push(scope);
    }

    // Group filter
    if (group === "core") {
      markers.push("not requires");
    } else if (group !== "all") {
      markers.push(group);
    }

    // Exclude slow tests unless explicitly requested
    if (!slow) {
      markers.push("not slow");
    }

    if (markers.length > 0) {
      args.push("-m", markers.join(" and "));
    }

    // Coverage
    if (coverage) {
      args.push("--cov", "--cov-report=term-missing");
    }

    // Verbose
    if (verbose) {
      args.push("-v");
    }

    // Stop on first failure
    if (exitfirst) {
      args.push("-x");
    }

    // Name filter (-k expression)
    if (filter) {
      args.push("-k", filter);
    }

    const result = Bun.spawnSync(args, {
      cwd,
      timeout: 600_000, // 10 minute timeout for e2e tests
    });

    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();

    return `Command: ${args.join(" ")}\nExit code: ${result.exitCode}\n\n${stdout}\n${stderr}`.trim();
  },
});
