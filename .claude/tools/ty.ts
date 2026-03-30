import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Run the Astral ty type checker on Python source files. Returns type errors and diagnostics.",
  args: z.object({
    paths: z
      .string()
      .optional()
      .describe("Specific file or directory to type-check (default: project root)"),
  }),
  async execute({ paths }) {
    const args = ["run", "ty", "check"];
    if (paths) args.push(paths);
    const result = Bun.spawnSync(["uv", ...args], {
      cwd: import.meta.dir + "/../..",
    });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
