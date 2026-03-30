import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Run the ruff linter on Python source files. Returns lint diagnostics and optionally auto-fixes issues.",
  args: z.object({
    paths: z
      .string()
      .default(".")
      .describe("File or directory paths to lint (default: current directory)"),
    fix: z
      .boolean()
      .default(true)
      .describe("Automatically fix fixable lint violations"),
  }),
  async execute({ paths, fix }) {
    const args = ["run", "ruff", "check"];
    if (fix) args.push("--fix");
    args.push(paths);
    const result = Bun.spawnSync(["uv", ...args], {
      cwd: import.meta.dir + "/../..",
    });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
